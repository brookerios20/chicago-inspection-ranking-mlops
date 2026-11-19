"""Tests for the ranking metrics and the capacity-based k."""

import numpy as np
import pandas as pd

from src.metrics import capacity_summary, precision_at_k, ranking_metrics, weekly_precision_at_k


def test_perfect_ranking():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.arange(10)[::-1]
    assert precision_at_k(y, scores, frac=0.2) == 1.0


def test_ties_are_broken_reproducibly():
    y = np.array([1, 0] * 50)
    scores = np.zeros(100)
    assert precision_at_k(y, scores, frac=0.1, seed=7) == precision_at_k(y, scores, frac=0.1, seed=7)


def test_weekly_precision_picks_top_k_in_each_week():
    # Week 1 (Mon 2025-01-06): 4 inspections, the 2 highest scores fail.
    # Week 2 (Mon 2025-01-13): 1 inspection, which passes; it is picked because the week has fewer than k.
    dates = pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-13"])
    y = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1, 0.5])
    precision, picked = weekly_precision_at_k(y, scores, pd.Series(dates), k=2)
    assert picked == 3
    assert np.isclose(precision, 2 / 3)


def test_capacity_k_is_share_of_median_weekly_canvass():
    # Three weeks with 10, 20, and 30 canvass inspections: median 20, so k = 2 at a 10% share.
    dates = pd.to_datetime(["2025-01-06"] * 10 + ["2025-01-13"] * 20 + ["2025-01-20"] * 30)
    train = pd.DataFrame({"inspection_date": dates, "fail": 0, "is_canvass": True})
    summary = capacity_summary(train, share=0.10)
    assert summary["median_weekly_canvass"] == 20
    assert summary["k_week"] == 2


def test_lift_is_relative_to_base_rate():
    dates = pd.Series(pd.to_datetime(["2025-01-06"] * 100))
    y = np.array([1] * 10 + [0] * 90)
    scores = np.r_[np.ones(10), np.zeros(90)]
    metrics = ranking_metrics(y, scores, dates, k_week=10)
    assert metrics["weekly_precision_at_k"] == 1.0
    assert metrics["precision_at_10"] == 1.0
    assert np.isclose(metrics["lift_per_100"], 90.0)
