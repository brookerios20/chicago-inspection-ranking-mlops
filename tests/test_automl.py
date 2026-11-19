"""Tests for the AutoML custom metric and the model assembly helpers."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src import config
from src.automl import make_metric
from src.clean import clean
from src.features import build_features
from src.train import assemble, feature_frame, fit_preprocessor, inner_split, refit


class FixedScores:
    """Stands in for a FLAML estimator: returns preset probabilities."""

    def __init__(self, scores):
        self.scores = np.asarray(scores)

    def predict_proba(self, X):
        return np.column_stack([1 - self.scores, self.scores])


def test_metric_returns_one_minus_weekly_precision():
    dates = pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-13"])
    y = np.array([1, 0, 0, 1])
    metric = make_metric(dates, k_week=1)
    loss, logged = metric(np.zeros((4, 1)), y, FixedScores([0.9, 0.1, 0.2, 0.8]), None, None, None)
    assert loss == 0.0  # the top pick in each week fails
    assert logged["weekly_precision_at_k"] == 1.0

    loss, _ = metric(np.zeros((4, 1)), y, FixedScores([0.1, 0.9, 0.2, 0.8]), None, None, None)
    assert np.isclose(loss, 0.5)  # week 1 picks a pass, week 2 picks a fail


def test_inner_split_stays_inside_the_training_window():
    dates = pd.to_datetime(["2023-10-29", "2023-10-30", "2024-03-01", "2024-03-02"])
    train = pd.DataFrame({"inspection_date": dates, "is_canvass": [True, True, True, False], "fail": 0})
    fit_rows, holdout = inner_split(train)
    assert fit_rows["inspection_date"].max() < config.INNER_VALID_START
    assert holdout["inspection_date"].min() >= config.INNER_VALID_START
    assert holdout["is_canvass"].all() and len(holdout) == 2  # the non-canvass row is left out
    assert config.FOOD_CODE_CHANGE < config.INNER_VALID_START < config.VALID_START


def test_feature_frame_fixes_types(split_raw):
    features = build_features(clean(split_raw))
    frame = feature_frame(features, "v2")
    assert list(frame.columns) == config.FEATURE_COLUMNS["v2"]
    assert all(frame[c].dtype == "float64" for c in config.NUMERIC_FEATURES["v2"])
    assert all(frame[c].map(type).eq(str).all() for c in config.CATEGORICAL_FEATURES)


def test_assembled_and_refit_models_score_raw_rows(split_raw):
    features = build_features(clean(split_raw))
    features.loc[features.index[:3], config.TARGET] = np.array([0, 1, 0], dtype="int8")  # both classes present
    preprocessor = fit_preprocessor(features, "v1")
    X = preprocessor.transform(feature_frame(features, "v1"))
    estimator = LogisticRegression(max_iter=200).fit(X, features[config.TARGET])

    model = assemble(preprocessor, estimator)
    scores = model.predict_proba(feature_frame(features, "v1"))[:, 1]
    assert scores.shape == (len(features),)

    refitted = refit(estimator, features, "v1")
    assert refitted.predict_proba(feature_frame(features, "v1")).shape == (len(features), 2)
