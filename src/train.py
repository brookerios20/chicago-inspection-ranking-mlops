"""Model assembly, evaluation, and refit, shared by the AutoML search and the registry step.

A model is an sklearn `Pipeline` of the shared preprocessing (`features.make_preprocessor`) and
an estimator. The preprocessing travels inside the logged model, so the deployed API receives
the raw feature columns and transforms them exactly as training did.

The model trains on every inspection type in the training window, with `inspection_group` as a
feature, and is evaluated on canvass inspections in the validation window, the population the
headline metric targets.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src import config
from src.features import make_preprocessor
from src.metrics import HEADLINE, ranking_metrics, slice_metrics, weekly_precision_at_k


def feature_frame(df: pd.DataFrame, version: str) -> pd.DataFrame:
    """The model's input columns for `version`: categories as text, numbers as floats.

    Fixing the types here gives the logged model a stable input signature, which the API's
    request schema copies.
    """
    out = df[config.FEATURE_COLUMNS[version]].copy()
    for col in config.CATEGORICAL_FEATURES:
        out[col] = out[col].astype(str)
    for col in config.NUMERIC_FEATURES[version]:
        out[col] = out[col].astype("float64")
    return out


def inner_split(train: pd.DataFrame, start: pd.Timestamp = config.INNER_VALID_START) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the training window for model selection.

    Returns the rows before `start`, which the AutoML search fits on, and the canvass rows from
    `start` on, which it selects by. Both lie inside the training window, so the validation
    window stays unseen until the promotion gate.
    """
    later = train["inspection_date"] >= start
    fit_rows = train[~later].reset_index(drop=True)
    holdout = train[later & train["is_canvass"]].reset_index(drop=True)
    return fit_rows, holdout


def fit_preprocessor(train: pd.DataFrame, version: str) -> ColumnTransformer:
    """Fit the shared preprocessing on training rows, with dense output for every estimator."""
    preprocessor = make_preprocessor(version).set_params(sparse_threshold=0.0)
    return preprocessor.fit(feature_frame(train, version))


def assemble(preprocessor: ColumnTransformer, estimator) -> Pipeline:
    """A model from an already fitted preprocessor and estimator."""
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def refit(estimator, rows: pd.DataFrame, version: str) -> Pipeline:
    """Refit a fresh copy of the preprocessing and of `estimator`'s configuration on `rows`."""
    model = Pipeline([
        ("preprocess", make_preprocessor(version).set_params(sparse_threshold=0.0)),
        ("model", clone(estimator)),
    ])
    return model.fit(feature_frame(rows, version), rows[config.TARGET])


def predict_scores(model: Pipeline, rows: pd.DataFrame, version: str) -> np.ndarray:
    """Predicted probability of Fail for each row."""
    return model.predict_proba(feature_frame(rows, version))[:, 1]


def evaluate(model: Pipeline, evaluate_rows: pd.DataFrame, version: str, k_week: int,
             seed: int = config.SEED) -> tuple[np.ndarray, dict, pd.DataFrame]:
    """Scores, ranking metrics, and slice metrics of `model` on `evaluate_rows`."""
    scores = predict_scores(model, evaluate_rows, version)
    metrics = ranking_metrics(evaluate_rows[config.TARGET], scores, evaluate_rows["inspection_date"], k_week, seed)
    slices = slice_metrics(evaluate_rows.assign(score=scores), "score", seed=seed)
    return scores, metrics, slices


def plot_precision_by_k(scores_by_name: dict, rows: pd.DataFrame, k_week: int, path: Path) -> Path:
    """Weekly precision@k for k from 1 to 3 times the capacity k, one line per model."""
    ks = np.arange(1, 3 * k_week + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, scores in scores_by_name.items():
        values = [weekly_precision_at_k(rows[config.TARGET], scores, rows["inspection_date"], k)[0] for k in ks]
        ax.plot(ks, values, label=name)
    ax.axhline(rows[config.TARGET].mean(), color="grey", linestyle=":", label="base rate")
    ax.axvline(k_week, color="grey", linestyle="--", label=f"capacity k = {k_week}")
    ax.set_xlabel("Inspections picked per week (k)")
    ax.set_ylabel(HEADLINE.replace("_", " "))
    ax.set_title("Weekly precision@k on canvass inspections, validation window")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_feature_importance(model: Pipeline, path: Path, top: int = 20) -> Path | None:
    """Bar chart of the estimator's feature importances or absolute coefficients, if it has either."""
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.ravel(estimator.coef_))
    else:
        return None
    names = model.named_steps["preprocess"].get_feature_names_out()
    order = np.argsort(values)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(order) + 1))
    ax.barh([names[i] for i in order], values[order])
    ax.set_title(f"Feature importance ({type(estimator).__name__})")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
