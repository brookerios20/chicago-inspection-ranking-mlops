"""Tests for the cleaning rules and the history and ZIP code features."""

import numpy as np
import pandas as pd
import pytest

from src.clean import clean, inspection_group
from src.features import build_features
from tests.conftest import make_raw


def by_id(features: pd.DataFrame) -> pd.DataFrame:
    return features.set_index("inspection_id")


def test_inspection_groups():
    assert inspection_group("canvass") == "canvass"
    assert inspection_group("canvass re-inspection") == "re-inspection"
    assert inspection_group("suspected food poisoning") == "complaint"
    assert inspection_group("license-task force") == "license"
    assert inspection_group(None) == "other"


def test_clean_drops_non_outcomes_and_unlinkable_licenses():
    raw = make_raw([
        {"inspection_id": "1", "license_number": "100", "inspection_date": "01/06/2020", "results": "Pass"},
        {"inspection_id": "2", "license_number": "100", "inspection_date": "01/07/2020", "results": "No Entry"},
        {"inspection_id": "3", "license_number": "0", "inspection_date": "01/08/2020", "results": "Fail"},
        {"inspection_id": "4", "license_number": None, "inspection_date": "01/09/2020", "results": "Fail"},
    ])
    assert clean(raw)["inspection_id"].tolist() == ["1"]


def test_restaurant_scope_keeps_only_restaurants():
    raw = make_raw([
        {"inspection_id": "1", "license_number": "100", "inspection_date": "01/06/2020", "results": "Pass"},
        {"inspection_id": "2", "license_number": "200", "inspection_date": "01/06/2020", "results": "Pass",
         "facility_type": "Grocery Store"},
        {"inspection_id": "3", "license_number": "300", "inspection_date": "01/06/2020", "results": "Pass",
         "facility_type": None},
        {"inspection_id": "4", "license_number": "400", "inspection_date": "01/06/2020", "results": "Pass",
         "facility_type": " RESTAURANT "},
    ])
    assert clean(raw, scope="all")["inspection_id"].tolist() == ["1", "2", "3", "4"]
    assert clean(raw, scope="restaurant")["inspection_id"].tolist() == ["1", "4"]


def test_unknown_scope_is_rejected():
    raw = make_raw([{"inspection_id": "1", "license_number": "100", "inspection_date": "01/06/2020",
                     "results": "Pass"}])
    with pytest.raises(ValueError, match="facility scope"):
        clean(raw, scope="bakeries")


def test_clean_maps_risk_and_missing_facility():
    raw = make_raw([
        {"inspection_id": "1", "license_number": "100", "inspection_date": "01/06/2020", "results": "Pass",
         "risk": "All", "facility_type": None},
    ])
    row = clean(raw, scope="all").iloc[0]  # the restaurant scope would drop a missing facility type
    assert row["risk"] == "missing"
    assert row["facility_group"] == "missing"


def test_first_inspections_have_no_history(same_day_raw):
    rows = by_id(build_features(clean(same_day_raw)))
    for inspection_id in ("1", "2"):
        row = rows.loc[inspection_id]
        assert row["n_prior"] == 0
        assert row["prior_fails"] == 0
        assert row["prev_result"] == "first inspection"
        assert np.isnan(row["days_since_prev"])
        assert np.isnan(row["prev_n_violations"])


def test_history_uses_earlier_days_only(same_day_raw):
    row = by_id(build_features(clean(same_day_raw))).loc["3"]
    assert row["n_prior"] == 2
    assert row["prior_fails"] == 1
    assert row["prev_result"] == "Pass"  # the later of the two same-day visits
    assert row["days_since_prev"] == 56
    assert row["prev_n_violations"] == 2


def test_prior_fails_match_brute_force():
    rng = np.random.default_rng(0)
    dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 60, 200), unit="D")
    raw = make_raw([
        {"inspection_id": str(i), "license_number": str(rng.integers(1, 6)),
         "inspection_date": d.strftime("%m/%d/%Y"), "results": "Fail" if rng.random() < 0.3 else "Pass"}
        for i, d in enumerate(dates)
    ])
    features = build_features(clean(raw))
    for row in features.itertuples():
        earlier = features[(features.license_number == row.license_number)
                           & (features.inspection_date < row.inspection_date)]
        assert row.prior_fails == earlier["fail"].sum()
        assert row.n_prior == len(earlier)


def test_zip_features_exclude_the_same_day():
    raw = make_raw([
        {"inspection_id": "1", "license_number": "100", "inspection_date": "01/06/2020", "results": "Fail",
         "inspection_type": "Complaint"},
        {"inspection_id": "2", "license_number": "200", "inspection_date": "01/06/2020", "results": "Pass"},
        {"inspection_id": "3", "license_number": "300", "inspection_date": "01/16/2020", "results": "Pass"},
    ])
    rows = by_id(build_features(clean(raw)))
    assert rows.loc["2", "zip_complaints_90d"] == 0
    assert rows.loc["3", "zip_complaints_90d"] == 1
    # Inspection 3 sees one earlier Fail among two inspections, so its rate moves above the prior.
    assert rows.loc["3", "zip_fail_rate_365d"] > rows.loc["2", "zip_fail_rate_365d"]
