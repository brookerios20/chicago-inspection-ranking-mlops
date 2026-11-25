"""Tests for src/governance.py: the fairness gate on the weekly picks and the privacy rule for logs."""

import numpy as np
import pandas as pd
import pytest

from src import governance


@pytest.fixture
def weekly() -> pd.DataFrame:
    """Four weeks of 20 inspections, half North and half South, with random scores and outcomes."""
    rng = np.random.default_rng(0)
    n = 80
    return pd.DataFrame({
        "inspection_date": pd.to_datetime("2026-01-05") + pd.to_timedelta(np.repeat(range(4), 20) * 7, unit="D"),
        "region": np.tile(["North", "South"], n // 2),
        "fail": rng.integers(0, 2, n),
        "prediction": rng.random(n),
    })


def test_weekly_picks_take_k_per_week(weekly):
    picked = governance.weekly_picks(weekly, "prediction", k=3)
    assert picked.groupby(weekly["inspection_date"]).sum().tolist() == [3, 3, 3, 3]


def test_gate_fails_a_ranking_that_picks_only_one_group(weekly):
    biased = weekly.assign(prediction=weekly["prediction"] + (weekly["region"] == "South"))
    passed, table, gaps = governance.fairness_gate(biased, "region", k=5)
    assert not passed
    assert table.loc["North", "selection rate"] == 0
    assert gaps["selection rate"] == pytest.approx(0.5)


def test_gate_passes_when_the_gaps_are_within_the_limit(weekly):
    passed, _, gaps = governance.fairness_gate(weekly, "region", k=5, max_gap=1.0)
    assert passed and set(gaps) == {"selection rate", "precision of picks"}


def test_privacy_rule_flags_names_and_addresses():
    assert governance.unloggable_fields({"facility_id", "risk", "dba_name", "address"}) == {"dba_name", "address"}
