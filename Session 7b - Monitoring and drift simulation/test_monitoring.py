"""Tests for src/monitoring.py: the request format, the drift scenarios and the alert thresholds.

The scenarios must corrupt a batch in the way their names say, and the input rules in
src/validate.py must flag the corruptions they are meant to catch.
"""

import numpy as np
import pandas as pd
import pytest

from src import metrics, monitoring, validate

VERSION = "v2"


@pytest.fixture
def batch() -> pd.DataFrame:
    """Two weeks of clean canvass inspections in the format of the processed splits."""
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame({
        "inspection_id": [str(1000 + i) for i in range(n)],
        "inspection_date": pd.to_datetime("2026-01-05") + pd.to_timedelta(rng.integers(0, 14, n), unit="D"),
        "fail": rng.integers(0, 2, n).astype("int8"),
        "inspection_group": "canvass", "facility_group": "restaurant",
        "risk": rng.choice(["high", "medium", "low"], n), "prev_result": rng.choice(["Pass", "Fail"], n),
        "prior_fails": rng.integers(0, 5, n).astype("int8"), "n_prior": rng.integers(5, 20, n),
        "days_since_prev": rng.uniform(100, 500, n), "prev_n_violations": rng.integers(0, 8, n).astype(float),
        "zip_fail_rate_365d": rng.uniform(0.1, 0.3, n), "zip_complaints_90d": rng.integers(0, 20, n).astype(float),
    })


def failed_rules(rows: pd.DataFrame) -> set[str]:
    report = validate.validate_features(rows, VERSION)
    return set(report.loc[~report["passed"], "rule"])


def test_clean_batch_passes_the_input_rules(batch):
    assert failed_rules(batch) == set()


def test_requests_carry_the_inspection_id_and_null_for_missing_values(batch):
    batch.loc[0, "days_since_prev"] = np.nan
    first = monitoring.to_requests(batch, ["days_since_prev", "risk"])[0]
    assert first == {"facility_id": "1000", "days_since_prev": None, "risk": batch.loc[0, "risk"]}


def test_out_of_bounds_values_break_the_range_rules(batch):
    corrupted = monitoring.out_of_bounds(batch, VERSION, share=0.5)
    assert failed_rules(corrupted) == {"range_prior_fails", "range_zip_fail_rate_365d"}


def test_swapped_columns_pass_the_input_rules_unnoticed(batch):
    swapped = monitoring.swap_columns(batch, VERSION)
    assert (swapped["prior_fails"] == batch["n_prior"]).all() and (swapped["n_prior"] == batch["prior_fails"]).all()
    assert failed_rules(swapped) == set()


def test_renamed_field_breaks_the_schema(batch):
    assert failed_rules(monitoring.rename_field(batch)) == {"required_feature_columns"}


def test_raw_risk_labels_are_unknown_categories(batch):
    assert failed_rules(monitoring.raw_risk_labels(batch)) == {"allowed_values_risk"}


def test_shuffled_outcomes_keep_each_weeks_fail_count(batch):
    shuffled = monitoring.shuffle_outcomes(batch)
    week = metrics.weeks(batch["inspection_date"])
    pd.testing.assert_series_equal(shuffled.groupby(week)["fail"].sum(), batch.groupby(week)["fail"].sum())
    assert not (shuffled["fail"] == batch["fail"]).all()


def test_thresholds_add_a_margin_to_clean_drift_and_keep_the_default_floor():
    clean = pd.Series({"drifted_share": 0.3, "zip_fail_rate_365d": 0.38, "risk": 0.01})
    assert monitoring.calibrate(clean) == {"zip_fail_rate_365d": 0.57, "risk": monitoring.DEFAULT_THRESHOLD}
