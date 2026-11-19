"""AutoML search with FLAML, scored by the project's headline metric.

FLAML tries several algorithms and hyperparameter settings within a time budget and keeps the
best configuration for each algorithm. The search uses a custom metric, weekly precision@k on
canvass inspections in the validation window, so it picks models by the same measure as the
baselines, the registry gate, and the presentation.

Inputs are the preprocessed matrices from `train.fit_preprocessor`, so the winning estimator
plugs into the same `Pipeline` the registry logs. FLAML trains each trial on the training rows
and scores it on the validation canvass rows (`eval_method="holdout"`), and the final refit on
training plus validation rows happens later in `train.refit`.
"""

import logging

import numpy as np
import pandas as pd
from flaml import AutoML

from src import config
from src.metrics import weekly_precision_at_k

logger = logging.getLogger(__name__)


def make_metric(dates_val, k_week: int, seed: int = config.SEED):
    """A FLAML custom metric that returns 1 minus weekly precision@k on the validation rows.

    `dates_val` holds the inspection dates of the validation rows, in the order FLAML receives
    them, so each row lands in its week. FLAML minimizes the returned loss.
    """
    dates_val = pd.Series(pd.to_datetime(np.asarray(dates_val)))

    def weekly_precision_loss(X_val, y_val, estimator, labels, X_train, y_train,
                              weight_val=None, weight_train=None, config=None,
                              groups_val=None, groups_train=None):
        scores = estimator.predict_proba(X_val)[:, 1]
        precision, picked = weekly_precision_at_k(y_val, scores, dates_val, k_week, seed)
        return 1.0 - precision, {"weekly_precision_at_k": precision, "picked": picked}

    return weekly_precision_loss


def run_search(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
               dates_val, k_week: int, time_budget: int = config.FLAML_TIME_BUDGET,
               estimators: list[str] = config.FLAML_ESTIMATORS, seed: int = config.SEED,
               log_file: str | None = None) -> AutoML:
    """Run the FLAML search and return the fitted `AutoML` object.

    `retrain_full=False` keeps the best model trained on the training rows only, so its
    validation score is out of sample. FLAML's own MLflow logging is off, and the pipeline logs
    one child run per algorithm instead.
    """
    automl = AutoML()
    automl.fit(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        task="classification",
        metric=make_metric(dates_val, k_week, seed),
        estimator_list=list(estimators),
        time_budget=time_budget,
        eval_method="holdout",
        retrain_full=False,
        seed=seed,
        n_jobs=-1,
        log_file_name=log_file or "",
        mlflow_logging=False,
        verbose=1,
    )
    logger.info("Best algorithm: %s (weekly precision@k %.3f)", automl.best_estimator, 1 - automl.best_loss)
    return automl


def per_estimator_summary(automl: AutoML) -> pd.DataFrame:
    """Best validation score and configuration found for each algorithm, best first."""
    rows = []
    for name, loss in automl.best_loss_per_estimator.items():
        if loss is None or not np.isfinite(loss):
            continue  # the algorithm finished no trial within the budget
        config_ = automl.best_config_per_estimator.get(name) or {}
        rows.append({
            "estimator": name,
            "weekly_precision_at_k": 1.0 - loss,
            # NumPy scalars become plain numbers and strings, so the CSV and MLflow show 0.05, not np.float64(0.05).
            "config": {k: (v.item() if hasattr(v, "item") else v) for k, v in config_.items()},
        })
    return pd.DataFrame(rows).sort_values("weekly_precision_at_k", ascending=False).reset_index(drop=True)


def best_estimator(automl: AutoML):
    """The fitted scikit-learn-compatible estimator behind FLAML's best model."""
    return automl.model.estimator
