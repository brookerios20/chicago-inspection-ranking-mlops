"""Data validation and verification rules, shared by the pipeline, the API, and monitoring.

Validation rules check structure: required columns, types, nulls, uniqueness, and allowed
values. A failed validation rule is an error, and `raise_on_errors` stops the pipeline.
Verification rules check whether structurally valid data is plausible against what the EDA
found (coordinate bounds, missing-value shares, fail rate, value ranges). A failed
verification rule is a warning that is logged and saved with the report.

Each check returns a report table with one row per rule: rule, level, passed, detail.
"""

import pandas as pd

from src import config

ERROR = "error"
WARNING = "warning"

# Highest missing share expected per raw column. The EDA found 28.2% for violations, 1.7% for
# facility type, 0.3% for coordinates, and under 0.1% for license and risk.
MAX_MISSING_SHARE = {
    "violations": 0.35,
    "facility_type": 0.03,
    "latitude": 0.01,
    "license_number": 0.01,
    "risk": 0.01,
}
FAIL_RATE_RANGE = (0.15, 0.30)  # yearly Fail rates in the EDA range from 0.196 to 0.259

# Plausible ranges for numeric features. Values outside them suggest corrupted input.
FEATURE_RANGES = {
    "prior_fails": (0, 200),
    "n_prior": (0, 400),
    "days_since_prev": (0, 6500),
    "prev_n_violations": (0, 60),
    "zip_fail_rate_365d": (0, 1),
    "zip_complaints_90d": (0, 2000),
}
ALLOWED_CATEGORIES = {
    "inspection_group": config.INSPECTION_GROUPS,
    "risk": config.RISK_LEVELS,
    "prev_result": config.PREV_RESULTS,
}


class DataValidationError(ValueError):
    """Raised when a validation rule fails."""


def _rule(rule: str, level: str, passed: bool, detail: str) -> dict:
    """One row of a report."""
    return {"rule": rule, "level": level, "passed": bool(passed), "detail": detail}


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Check the raw snapshot written by ingest."""
    rows = []
    missing = [c for c in config.RAW_COLUMNS if c not in df.columns]
    rows.append(_rule("required_columns", ERROR, not missing, f"missing: {missing}" if missing else "all present"))
    if missing:
        return pd.DataFrame(rows)  # the remaining rules need these columns

    # Validation: structure and allowed values.
    ids = df["inspection_id"]
    rows.append(_rule("inspection_id_not_null", ERROR, ids.notna().all(), f"{ids.isna().sum()} null"))
    rows.append(_rule("inspection_id_unique", ERROR, not ids.duplicated().any(),
                      f"{ids.duplicated().sum()} duplicated"))

    dates = df["inspection_date"]
    is_date = pd.api.types.is_datetime64_any_dtype(dates)
    rows.append(_rule("inspection_date_is_datetime", ERROR, is_date, str(dates.dtype)))
    if is_date:
        rows.append(_rule("inspection_date_not_null", ERROR, dates.notna().all(), f"{dates.isna().sum()} null"))
        in_range = dates.dropna().between(config.EARLIEST_DATE, config.SNAPSHOT_END).all()
        rows.append(_rule("inspection_date_in_snapshot", ERROR, in_range, f"{dates.min()} to {dates.max()}"))

    results = df["results"]
    unknown = sorted(set(results.dropna()) - set(config.ALL_RESULTS))
    rows.append(_rule("results_not_null", ERROR, results.notna().all(), f"{results.isna().sum()} null"))
    rows.append(_rule("results_allowed_values", ERROR, not unknown, f"unknown: {unknown}" if unknown else "ok"))

    numeric = all(pd.api.types.is_numeric_dtype(df[c]) for c in ("latitude", "longitude"))
    rows.append(_rule("coordinates_numeric", ERROR, numeric, f"{df['latitude'].dtype}, {df['longitude'].dtype}"))

    # Verification: plausibility against the EDA.
    if numeric:
        lat, lon = df["latitude"], df["longitude"]
        outside = (lat.notna() & ~lat.between(*config.LAT_BOUNDS)) | (lon.notna() & ~lon.between(*config.LON_BOUNDS))
        rows.append(_rule("coordinates_in_chicago", WARNING, not outside.any(), f"{outside.sum()} outside"))

    for col, limit in MAX_MISSING_SHARE.items():
        share = df[col].isna().mean()
        rows.append(_rule(f"missing_share_{col}", WARNING, share <= limit, f"{share:.3f} (limit {limit})"))

    modeling = results.isin(config.MODEL_RESULTS)
    fail_rate = (results[modeling] == "Fail").mean() if modeling.any() else float("nan")
    low, high = FAIL_RATE_RANGE
    rows.append(_rule("fail_rate_in_range", WARNING, low <= fail_rate <= high, f"{fail_rate:.3f} (range {low} to {high})"))

    if is_date and dates.notna().any():
        months = dates.dt.to_period("M")
        expected = pd.period_range(months.min(), months.max(), freq="M")
        empty = sorted(str(m) for m in set(expected) - set(months.dropna()))
        rows.append(_rule("no_empty_months", WARNING, not empty, f"empty: {empty}" if empty else "ok"))

    return pd.DataFrame(rows)


def validate_features(df: pd.DataFrame, version: str = config.FEATURE_VERSION,
                      require_target: bool = False) -> pd.DataFrame:
    """Check a feature table, or a batch of API requests, against the model's input schema."""
    required = list(config.FEATURE_COLUMNS[version]) + ([config.TARGET] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    rows = [_rule("required_feature_columns", ERROR, not missing, f"missing: {missing}" if missing else "all present")]
    if missing:
        return pd.DataFrame(rows)

    numeric_cols = config.NUMERIC_FEATURES[version]
    not_numeric = [c for c in numeric_cols if not pd.api.types.is_numeric_dtype(df[c])]
    rows.append(_rule("numeric_feature_types", ERROR, not not_numeric,
                      f"not numeric: {not_numeric}" if not_numeric else "ok"))
    null_cats = [c for c in config.CATEGORICAL_FEATURES if df[c].isna().any()]
    rows.append(_rule("categorical_not_null", ERROR, not null_cats, f"null in: {null_cats}" if null_cats else "ok"))

    for col, allowed in ALLOWED_CATEGORIES.items():
        unknown = sorted(set(df[col].dropna().astype(str)) - set(allowed))
        rows.append(_rule(f"allowed_values_{col}", WARNING, not unknown, f"unknown: {unknown}" if unknown else "ok"))

    for col in numeric_cols:
        if col in not_numeric:
            continue
        low, high = FEATURE_RANGES[col]
        values = df[col].dropna()
        outside = int((~values.between(low, high)).sum())
        rows.append(_rule(f"range_{col}", WARNING, outside == 0, f"{outside} outside {low} to {high}"))

    return pd.DataFrame(rows)


def raise_on_errors(report: pd.DataFrame, stage: str) -> None:
    """Stop the pipeline when any validation rule in `report` failed."""
    failed = report[(report["level"] == ERROR) & ~report["passed"]]
    if not failed.empty:
        details = "; ".join(f"{r.rule} ({r.detail})" for r in failed.itertuples())
        raise DataValidationError(f"{stage} validation failed: {details}")


def warnings_of(report: pd.DataFrame) -> pd.DataFrame:
    """The verification rules in `report` that did not pass."""
    return report[(report["level"] == WARNING) & ~report["passed"]]
