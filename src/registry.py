"""Promotion gate, semantic versioning, and registration in the MLflow Model Registry.

Gate (champion and challenger). A challenger is registered only if its validation headline
beats two bars, each raised by `config.GATE_SE_MULTIPLIER` standard errors:

- the best baseline in results/baselines.csv
- the current production model's validation headline, recorded on its registry version when it
  was registered. The production model was refit on the validation rows, so scoring it on them
  again would overstate it; the recorded out-of-sample score is the fair comparison.

Semantic versions. MLflow numbers registry versions 1, 2, 3; each version also carries a
`semver` tag computed from what changed since the production version:

- MAJOR: the input signature changed (features added, removed, or retyped), which breaks the
  deployed API's request schema
- MINOR: same signature, but a new data version or a different algorithm
- PATCH: same signature, data, and algorithm, with new hyperparameters or code

The `production` alias points at the newest version that passed the gate, and the API loads
`models:/inspection-ranker@production`. Before any challenger has passed, the best logistic
regression baseline is registered as the first production version (`promotion_action`), so the
deployment stage has a model and every later challenger is judged against it.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import skops.io
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature

from src import config, tracking
from src.metrics import HEADLINE


@dataclass
class Bar:
    """A score the challenger must beat, with its standard error and where it came from."""
    name: str
    headline: float
    se: float

    @property
    def threshold(self) -> float:
        return self.headline + config.GATE_SE_MULTIPLIER * self.se


def best_baseline(path: Path = config.RESULTS_DIR / "baselines.csv") -> Bar:
    """The baseline with the highest headline in the baselines table."""
    table = pd.read_csv(path).sort_values(HEADLINE, ascending=False)
    top = table.iloc[0]
    return Bar(f"baseline {top['baseline']}", float(top[HEADLINE]), float(top["se_headline"]))


def production_version():
    """The registry version behind the production alias, or None before the first registration."""
    try:
        return tracking.client().get_model_version_by_alias(config.MODEL_NAME, config.PRODUCTION_ALIAS)
    except MlflowException:
        return None


def production_bar(version) -> Bar | None:
    """The production model's recorded validation headline, if it has one."""
    if version is None or "valid_headline" not in version.tags:
        return None
    return Bar(f"production v{version.tags.get('semver', version.version)}",
               float(version.tags["valid_headline"]), float(version.tags["valid_se"]))


def gate(challenger: float, bars: list[Bar | None]) -> tuple[bool, str]:
    """Whether `challenger` beats every bar's threshold, and a one-line reason."""
    bars = [b for b in bars if b is not None]
    failed = [b for b in bars if challenger <= b.threshold]
    if failed:
        worst = max(failed, key=lambda b: b.threshold)
        return False, (f"challenger {challenger:.3f} does not beat {worst.name} "
                       f"({worst.headline:.3f} + {config.GATE_SE_MULTIPLIER:g} SE = {worst.threshold:.3f})")
    beaten = ", ".join(f"{b.name} ({b.threshold:.3f})" for b in bars)
    return True, f"challenger {challenger:.3f} beats {beaten}"


def promotion_action(gate_passed: bool, has_production: bool) -> str:
    """What to register after the gate.

    - "challenger": the AutoML challenger passed the gate
    - "bootstrap_baseline": the challenger failed and no production model exists yet, so the best
      model-based baseline (a logistic regression) becomes the first production version, and
      later challengers must beat it
    - "none": the challenger failed and a production model already serves
    """
    if gate_passed:
        return "challenger"
    return "none" if has_production else "bootstrap_baseline"


def signature_hash(version: str) -> str:
    """Short hash of the input schema (column names and types) for feature `version`."""
    schema = [(col, "str" if col in config.CATEGORICAL_FEATURES else "float64")
              for col in config.FEATURE_COLUMNS[version]]
    return hashlib.sha256(json.dumps(schema).encode()).hexdigest()[:12]


def bump_semver(previous: dict | None, current: dict) -> str:
    """Next semantic version, from the previous production version's tags and the current ones.

    Both dictionaries hold `signature`, `data_processed_md5`, and `estimator`; `previous` also
    holds `semver`. With no previous version, the result is 1.0.0.
    """
    if not previous or "semver" not in previous:
        return "1.0.0"
    major, minor, patch = (int(part) for part in previous["semver"].split("."))
    if previous.get("signature") != current["signature"]:
        return f"{major + 1}.0.0"
    if (previous.get("data_processed_md5") != current["data_processed_md5"]
            or previous.get("estimator") != current["estimator"]):
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def trusted_types(model) -> list[str]:
    """Types in `model` that skops does not trust by default, such as `numpy.dtype`.

    MLflow saves scikit-learn models with skops, which refuses to reload a file holding types it
    has not been told are safe. These types come from a model this pipeline just built (the
    preprocessing's encoder dtype, or a LightGBM or XGBoost estimator), so they are trusted.
    """
    return skops.io.get_untrusted_types(data=skops.io.dumps(model))


def register(run_id: str, model, X_example: pd.DataFrame, tags: dict, description: str) -> tuple[str, str]:
    """Log `model` to `run_id`, register it, tag it, and point the production alias at it.

    `tags` holds at least signature, data_processed_md5, estimator, valid_headline, and
    valid_se. Returns the registry version number and its semantic version.
    """
    previous = production_version()
    semver = bump_semver(dict(previous.tags) if previous else None, tags)

    signature = infer_signature(X_example, model.predict_proba(X_example))
    with tracking.active_run(run_id):
        info = mlflow.sklearn.log_model(
            model, name="model", signature=signature, input_example=X_example.head(5),
            pyfunc_predict_fn="predict_proba", registered_model_name=config.MODEL_NAME,
            skops_trusted_types=trusted_types(model),
        )
    version = str(info.registered_model_version)

    c = tracking.client()
    for key, value in {**tags, "semver": semver}.items():
        c.set_model_version_tag(config.MODEL_NAME, version, key, str(value))
    c.update_model_version(config.MODEL_NAME, version, description=f"{semver}: {description}")
    c.set_registered_model_alias(config.MODEL_NAME, config.PRODUCTION_ALIAS, version)
    return version, semver
