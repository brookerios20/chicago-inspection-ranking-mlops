"""Stage 5: score simple baselines on the validation window.

The baselines set the bar a trained model must clear before it is registered. All of them are
scored on canvass inspections in the validation window, the population the metric targets.

- random: a random ranking, whose precision@k sits near the base rate
- rate:<column>: each inspection gets the Fail rate of its category among canvass inspections
  in the training window, as in the EDA notebook
- logistic_regression_<version>: a logistic regression on the v1 or v2 features, trained on
  all training rows, the same population the AutoML search uses

Results go to `results/baselines.csv` (one row per baseline), `results/baselines_slices.csv`
(precision@k per slice), and `results/baselines.md` (a summary table).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.features import make_preprocessor
from src.metrics import k_name, ranking_metrics, slice_metrics

logger = logging.getLogger(__name__)

RATE_COLUMNS = ["risk", "facility_group", "prev_result", "prior_fails_capped", "zip"]


def _with_capped_fails(df: pd.DataFrame) -> pd.DataFrame:
    """Add earlier fails capped at 5, so rare high counts share one category."""
    return df.assign(prior_fails_capped=df["prior_fails"].clip(upper=5))


def rate_scores(fit: pd.DataFrame, evaluate: pd.DataFrame, column: str) -> np.ndarray:
    """Score each row with the Fail rate of its category in `fit`, or the overall rate if unseen."""
    rates = fit.groupby(column)[config.TARGET].mean()
    return evaluate[column].map(rates).astype(float).fillna(fit[config.TARGET].mean()).to_numpy()


def fit_logistic(train: pd.DataFrame, version: str) -> Pipeline:
    """Logistic regression on feature `version`, with the shared preprocessing."""
    model = Pipeline([
        ("preprocess", make_preprocessor(version)),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    return model.fit(train[config.FEATURE_COLUMNS[version]], train[config.TARGET])


def baseline_scores(train: pd.DataFrame, evaluate: pd.DataFrame, seed: int = config.SEED) -> dict:
    """Scores for every baseline on the rows of `evaluate`."""
    train, evaluate = _with_capped_fails(train), _with_capped_fails(evaluate)
    canvass_fit = train[train["is_canvass"]]
    scores = {"random": np.random.default_rng(seed).random(len(evaluate))}
    for column in RATE_COLUMNS:
        scores[f"rate:{column}"] = rate_scores(canvass_fit, evaluate, column)
    for version in config.FEATURE_COLUMNS:
        model = fit_logistic(train, version)
        scores[f"logistic_regression_{version}"] = model.predict_proba(evaluate[config.FEATURE_COLUMNS[version]])[:, 1]
    return scores


def _markdown_table(df: pd.DataFrame) -> str:
    """Render a table as GitHub markdown, with floats to three decimals."""
    def fmt(value):
        return f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def run_baselines(train: pd.DataFrame, valid: pd.DataFrame, out_dir: Path = config.RESULTS_DIR,
                  seed: int = config.SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every baseline on canvass validation rows, write the results, and return them."""
    evaluate = valid[valid["is_canvass"]].reset_index(drop=True)
    scores = baseline_scores(train, evaluate, seed)

    table = pd.DataFrame([{"baseline": name, **ranking_metrics(evaluate[config.TARGET], s, seed)}
                          for name, s in scores.items()])
    headline = k_name(config.HEADLINE_K)
    table = table.sort_values(headline, ascending=False).reset_index(drop=True)

    slices = pd.concat(
        [slice_metrics(evaluate.assign(score=s), "score", seed=seed).assign(baseline=name)
         for name, s in scores.items()],
        ignore_index=True,
    )
    slices = slices[["baseline"] + [c for c in slices.columns if c != "baseline"]]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "baselines.csv", index=False)
    slices.to_csv(out_dir / "baselines_slices.csv", index=False)

    summary_cols = ["baseline"] + [k_name(f) for f in config.K_FRACS] + ["se_at_headline", "lift_per_100",
                                                                          "roc_auc", "pr_auc"]
    header = (
        "# Baselines\n\n"
        f"Scored on {len(evaluate):,} canvass inspections in the validation window "
        f"({evaluate['inspection_date'].min().date()} to {evaluate['inspection_date'].max().date()}), "
        f"with a Fail base rate of {evaluate[config.TARGET].mean():.3f}. Rate baselines use category Fail "
        f"rates from {int(train['is_canvass'].sum()):,} canvass inspections in the training window, and the "
        f"logistic regressions train on all {len(train):,} training inspections. The headline metric is "
        f"{headline}, and se_at_headline is its standard error.\n\n"
    )
    (out_dir / "baselines.md").write_text(header + _markdown_table(table[summary_cols]) + "\n")
    logger.info("Best baseline: %s (%s %.3f)", table.loc[0, "baseline"], headline, table.loc[0, headline])
    return table, slices
