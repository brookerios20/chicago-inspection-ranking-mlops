"""Stage 3: build the model features and the shared preprocessing transformer.

Every feature for an inspection uses only inspections on strictly earlier days. Same-day
visits are excluded because a re-inspection can follow a failure within hours, and its result
would not be known when the first visit is scheduled. The same rule holds for the API and the
monitoring stage, because they receive rows built by this module.

Feature version v1 uses the establishment's own history. Version v2 adds two neighborhood
features for the establishment's ZIP code. `build_features` computes both, and
`config.FEATURE_COLUMNS[version]` selects the columns a model uses.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

KEYS = ["license_number", "inspection_date"]


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the establishment's inspection history as of the start of each inspection day.

    - n_prior: modeling inspections on earlier days
    - prior_fails: Fail results on earlier days
    - prev_result: result of the last inspection on the most recent earlier day
    - days_since_prev: days since that day
    - prev_n_violations: violations cited at that inspection
    """
    ordered = df.sort_values(KEYS + ["inspection_id_num"], kind="mergesort")
    day = (
        ordered.groupby(KEYS, sort=True)
        .agg(day_n=(config.TARGET, "size"), day_fails=(config.TARGET, "sum"),
             day_last_result=("results", "last"), day_last_n_violations=("n_violations", "last"))
        .reset_index()
    )
    by_license = day.groupby("license_number", sort=False)
    # Cumulative totals minus the current day give totals over earlier days only.
    day["n_prior"] = by_license["day_n"].cumsum() - day["day_n"]
    day["prior_fails"] = by_license["day_fails"].cumsum() - day["day_fails"]
    day["prev_result"] = by_license["day_last_result"].shift(1)
    day["prev_n_violations"] = by_license["day_last_n_violations"].shift(1).astype(float)
    day["days_since_prev"] = (day["inspection_date"] - by_license["inspection_date"].shift(1)).dt.days.astype(float)

    history = day[KEYS + ["n_prior", "prior_fails", "prev_result", "days_since_prev", "prev_n_violations"]]
    out = df.merge(history, on=KEYS, how="left", validate="many_to_one")
    out["prev_result"] = out["prev_result"].fillna("first inspection").astype(str)
    return out


def _trailing_zip_sums(df: pd.DataFrame, value_cols: list[str], window_days: int) -> pd.DataFrame:
    """Sum `value_cols` per ZIP code over the `window_days` days before each date, excluding that date."""
    daily = df.dropna(subset=["zip"]).groupby(["zip", "inspection_date"])[value_cols].sum().sort_index()
    rolled = (
        daily.reset_index(level="zip")
        .groupby("zip")[value_cols]
        .rolling(f"{window_days}D", closed="left")
        .sum()
    )
    return rolled.fillna(0.0)


def add_zip_features(df: pd.DataFrame, prior_rate: float | None = None) -> pd.DataFrame:
    """Add the ZIP code's recent fail rate and complaint count, from earlier days only.

    - zip_fail_rate_365d: Fail share of the ZIP code's inspections in the previous 365 days,
      smoothed toward `prior_rate` so ZIP codes with few inspections stay near it
    - zip_complaints_90d: complaint inspections in the ZIP code in the previous 90 days

    `prior_rate` defaults to the Fail rate before the validation window. Rows without a valid
    ZIP code get the prior rate and zero complaints.
    """
    if prior_rate is None:
        prior_rate = float(df.loc[df["inspection_date"] < config.VALID_START, config.TARGET].mean())
    counts = df.assign(n=1, complaint=df["inspection_group"].eq("complaint").astype(int))

    rate = _trailing_zip_sums(counts, ["n", config.TARGET], config.ZIP_RATE_WINDOW_DAYS)
    complaints = _trailing_zip_sums(counts, ["complaint"], config.COMPLAINT_WINDOW_DAYS)
    trailing = rate.join(complaints).reset_index()

    m = config.ZIP_RATE_SMOOTHING
    trailing["zip_fail_rate_365d"] = (trailing[config.TARGET] + m * prior_rate) / (trailing["n"] + m)
    trailing["zip_complaints_90d"] = trailing["complaint"]

    out = df.merge(trailing[["zip", "inspection_date", "zip_fail_rate_365d", "zip_complaints_90d"]],
                   on=["zip", "inspection_date"], how="left", validate="many_to_one")
    out["zip_fail_rate_365d"] = out["zip_fail_rate_365d"].fillna(prior_rate)
    out["zip_complaints_90d"] = out["zip_complaints_90d"].fillna(0.0)
    return out


def build_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Add every feature to the cleaned rows and return one row per inspection."""
    out = add_zip_features(add_history_features(cleaned))
    meta = ["inspection_id", "license_number", "inspection_date", config.TARGET, "is_canvass", "zip",
            "latitude", "longitude"]
    features = config.CATEGORICAL_FEATURES + config.NUMERIC_FEATURES["v2"]
    return out[meta + features].reset_index(drop=True)


def make_preprocessor(version: str = config.FEATURE_VERSION) -> ColumnTransformer:
    """One-hot encode categories and impute and scale numbers for feature `version`.

    Unseen categories at scoring time encode as all zeros. Missing numeric values, such as the
    days since the previous inspection for a first inspection, are filled with the training
    median and flagged by an indicator column.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
        ("numeric", numeric, config.NUMERIC_FEATURES[version]),
    ])
