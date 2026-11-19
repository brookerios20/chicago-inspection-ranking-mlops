"""Tests for the ranking metrics."""

import numpy as np

from src.metrics import precision_at_k, ranking_metrics


def test_perfect_ranking():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.arange(10)[::-1]
    assert precision_at_k(y, scores, frac=0.2) == 1.0


def test_ties_are_broken_reproducibly():
    y = np.array([1, 0] * 50)
    scores = np.zeros(100)
    assert precision_at_k(y, scores, frac=0.1, seed=7) == precision_at_k(y, scores, frac=0.1, seed=7)


def test_lift_is_relative_to_base_rate():
    y = np.array([1] * 10 + [0] * 90)
    scores = np.r_[np.ones(10), np.zeros(90)]
    metrics = ranking_metrics(y, scores)
    assert metrics["precision_at_10"] == 1.0
    assert np.isclose(metrics["lift_per_100"], 90.0)
