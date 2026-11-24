"""Stage 5: score simple baselines on the validation window.

The baselines set the bar a trained model must clear before it is registered. All of them are
scored on canvass inspections in the validation window, the population the metric targets,
with the headline weekly precision@k (k from `metrics.capacity_summary` on the training window).

- random: a random ranking, whose precision sits near the base rate
- rate:<column>: each inspection gets the Fail rate of its category among canvass inspections
  in the training window, as in the EDA notebook. A column with one value in the scope (such as
  facility type for restaurants only) is skipped.
- logistic_regression_<version>: a logistic regression on the v1 or v2 features, trained on
  all training rows, the same population the AutoML search uses

Outputs, all in `results/`:

- `baselines.csv`: one row per baseline with every metric
- `baselines_slices.csv`: precision@10% per slice
- `baselines.md`: a summary table
- `capacity.json`: weekly inspection counts in the training window and the k they imply
- `metrics.json`: the headline per baseline, which `dvc metrics diff` compares across data versions
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.features import make_preprocessor
from src.metrics import HEADLINE, capacity_summary, k_name, ranking_metrics, slice_metrics

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
        if canvass_fit[column].nunique() < 2:
            continue
        scores[f"rate:{column}"] = rate_scores(canvass_fit, evaluate, column)
    for version in config.FEATURE_COLUMNS:
        model = fit_logistic(train, version)
        scores[f"logistic_regression_{version}"] = model.predict_proba(evaluate[config.FEATURE_COLUMNS[version]])[:, 1]
    return scores


def markdown_table(df: pd.DataFrame) -> str:
    """Render a table as GitHub markdown, with floats to three decimals."""
    def fmt(value):
        return f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _write_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def run_baselines(train: pd.DataFrame, valid: pd.DataFrame, out_dir: Path = config.RESULTS_DIR,
                  seed: int = config.SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every baseline on canvass validation rows, write the results, and return them."""
    capacity = capacity_summary(train)
    k_week = capacity["k_week"]
    evaluate = valid[valid["is_canvass"]].reset_index(drop=True)
    scores = baseline_scores(train, evaluate, seed)

    table = pd.DataFrame([
        {"baseline": name, **ranking_metrics(evaluate[config.TARGET], s, evaluate["inspection_date"], k_week, seed)}
        for name, s in scores.items()
    ])
    table = table.sort_values(HEADLINE, ascending=False).reset_index(drop=True)

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
    _write_json({"facility_scope": config.FACILITY_SCOPE, **capacity}, out_dir / "capacity.json")
    _write_json({
        "facility_scope": config.FACILITY_SCOPE,
        "k_week": k_week,
        "baselines": {row.baseline: {HEADLINE: round(getattr(row, HEADLINE), 4),
                                     "lift_per_100": round(row.lift_per_100, 2),
                                     "roc_auc": round(row.roc_auc, 4)}
                      for row in table.itertuples()},
    }, out_dir / "metrics.json")

    summary_cols = ["baseline", HEADLINE, "se_headline", "lift_per_100"] + [k_name(f) for f in config.K_FRACS] + [
        "roc_auc", "pr_auc"]
    header = (
        "# Baselines\n\n"
        f"Facility scope: {config.FACILITY_SCOPE}. Scored on {len(evaluate):,} canvass inspections in the "
        f"validation window ({evaluate['inspection_date'].min().date()} to "
        f"{evaluate['inspection_date'].max().date()}), with a Fail base rate of "
        f"{evaluate[config.TARGET].mean():.3f}.\n\n"
        f"The headline, {HEADLINE}, ranks each week's canvass inspections and scores the top {k_week}. "
        f"k is {config.CAPACITY_SHARE:.0%} of the median {capacity['median_weekly_canvass']:.0f} canvass "
        f"inspections per week in the training window, where the median for all inspection types is "
        f"{capacity['median_weekly_inspections']:.0f} per week. se_headline is its standard error, and the "
        f"precision_at columns rank the whole window at once.\n\n"
        f"Rate baselines use category Fail rates from {int(train['is_canvass'].sum()):,} canvass inspections "
        f"in the training window, and the logistic regressions train on all {len(train):,} training "
        f"inspections.\n\n"
    )
    (out_dir / "baselines.md").write_text(header + markdown_table(table[summary_cols]) + "\n")
    logger.info("k_week = %d (median %.0f canvass inspections per week)", k_week, capacity["median_weekly_canvass"])
    logger.info("Best baseline: %s (%s %.3f)", table.loc[0, "baseline"], HEADLINE, table.loc[0, HEADLINE])
    return table, slices
