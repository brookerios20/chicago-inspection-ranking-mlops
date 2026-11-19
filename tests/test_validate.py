"""Tests for the validation (hard-fail) and verification (warning) rules."""

import pandas as pd
import pytest

from src.clean import clean
from src.features import build_features
from src.validate import DataValidationError, raise_on_errors, validate_features, validate_raw, warnings_of


def failed_errors(report: pd.DataFrame) -> set[str]:
    return set(report.loc[(report["level"] == "error") & ~report["passed"], "rule"])


def test_clean_snapshot_has_no_errors(split_raw):
    raise_on_errors(validate_raw(split_raw), "raw")


@pytest.mark.parametrize("rule, corrupt", [
    ("required_columns", lambda df: df.drop(columns=["results"])),
    ("inspection_id_unique", lambda df: df.assign(inspection_id="1")),
    ("inspection_id_not_null", lambda df: df.assign(inspection_id=pd.NA)),
    ("inspection_date_is_datetime", lambda df: df.assign(inspection_date=df["inspection_date"].astype(str))),
    ("inspection_date_in_snapshot", lambda df: df.assign(inspection_date=pd.Timestamp("2030-01-01"))),
    ("results_allowed_values", lambda df: df.assign(results="Maybe")),
    ("coordinates_numeric", lambda df: df.assign(latitude="north")),
])
def test_each_error_rule_stops_the_pipeline(split_raw, rule, corrupt):
    report = validate_raw(corrupt(split_raw.copy()))
    assert rule in failed_errors(report)
    with pytest.raises(DataValidationError):
        raise_on_errors(report, "raw")


def test_out_of_bounds_coordinates_warn(split_raw):
    report = validate_raw(split_raw.assign(latitude=45.0))
    assert not failed_errors(report)
    assert "coordinates_in_chicago" in set(warnings_of(report)["rule"])


def test_feature_schema_errors_and_range_warnings(split_raw):
    features = build_features(clean(split_raw))
    assert not failed_errors(validate_features(features, "v2"))

    missing = validate_features(features.drop(columns=["prior_fails"]), "v1")
    assert "required_feature_columns" in failed_errors(missing)

    retyped = validate_features(features.assign(prior_fails="many"), "v1")
    assert "numeric_feature_types" in failed_errors(retyped)

    out_of_range = validate_features(features.assign(days_since_prev=-5.0), "v1")
    assert not failed_errors(out_of_range)
    assert "range_days_since_prev" in set(warnings_of(out_of_range)["rule"])
