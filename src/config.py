"""Project settings: paths, the data snapshot, split dates, metric settings, and feature lists.

Every module reads its settings from here, so a change made in this file applies to the whole
pipeline. Values a reader is likely to change can be overridden with the environment variable
named in the comment beside them. Dates use the ISO format YYYY-MM-DD.
"""

import os
from pathlib import Path

import pandas as pd


def _env_path(name: str, default: Path) -> Path:
    """Return the path in environment variable `name`, or `default` when it is unset."""
    return Path(os.getenv(name, str(default)))


def _env_date(name: str, default: str) -> pd.Timestamp:
    """Return the date in environment variable `name`, or `default` when it is unset."""
    return pd.Timestamp(os.getenv(name, default))


# --- Paths ------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = _env_path("INSPECTIONS_DATA_DIR", PROJECT_ROOT / "data")  # INSPECTIONS_DATA_DIR
RAW_CSV = _env_path("INSPECTIONS_RAW_CSV", DATA_DIR / "Food_Inspections_20261102.csv")  # INSPECTIONS_RAW_CSV
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = _env_path("INSPECTIONS_RESULTS_DIR", PROJECT_ROOT / "results")  # INSPECTIONS_RESULTS_DIR

RAW_PARQUET = RAW_DIR / "inspections.parquet"
RAW_MANIFEST = RAW_DIR / "manifest.json"
SPLIT_MANIFEST_NAME = "split_manifest.json"

# --- Source -----------------------------------------------------------------------------------

DATASET_ID = "4ijn-s7e5"
API_URL = f"https://data.cityofchicago.org/resource/{DATASET_ID}.csv"
API_PAGE_SIZE = 50_000

# Columns kept from the export, in snake_case. The export's Location column repeats latitude
# and longitude as text, so it is dropped.
RAW_COLUMNS = [
    "inspection_id", "dba_name", "aka_name", "license_number", "facility_type", "risk",
    "address", "city", "state", "zip", "inspection_date", "inspection_type", "results",
    "violations", "latitude", "longitude",
]

# --- Snapshot and split dates -----------------------------------------------------------------

# The last inspection date included. Fixing it means a later export yields the same rows,
# apart from records the city revises after the fact, which the manifest hash reveals.
SNAPSHOT_END = _env_date("SNAPSHOT_END", "2026-10-29")  # SNAPSHOT_END
EARLIEST_DATE = pd.Timestamp("2010-01-01")

# Chicago adopted the FDA Food Code on this date, which renumbered the violation codes.
# Training starts here so that violation-based history uses one coding scheme.
FOOD_CODE_CHANGE = pd.Timestamp("2018-07-01")
VALID_START = _env_date("VALID_START", "2024-10-30")  # VALID_START
TEST_START = _env_date("TEST_START", "2025-10-30")  # TEST_START

# drift_pre2018: before FOOD_CODE_CHANGE, kept for the drift demonstration.
# train: FOOD_CODE_CHANGE up to the day before VALID_START.
# valid: VALID_START up to the day before TEST_START, used for baselines and model selection.
# test: TEST_START through SNAPSHOT_END, held out until production validation.
SPLITS = ("train", "valid", "test", "drift_pre2018")

# --- Evaluation -------------------------------------------------------------------------------

SEED = int(os.getenv("SEED", "42"))  # SEED
K_FRACS = (0.05, 0.10, 0.20)  # shares of ranked canvass inspections scored by precision@k
HEADLINE_K = 0.10

# Chicago's North and South Sides divide at Roosevelt Road, near this latitude.
ROOSEVELT_LAT = 41.8675
SLICE_FACILITIES = ("restaurant", "grocery store", "school")

# --- Cleaning ---------------------------------------------------------------------------------

MODEL_RESULTS = ("Pass", "Pass w/ Conditions", "Fail")
ALL_RESULTS = MODEL_RESULTS + ("Out of Business", "No Entry", "Not Ready", "Business Not Located")
INSPECTION_GROUPS = ("canvass", "complaint", "license", "re-inspection", "other")
RISK_LEVELS = ("high", "medium", "low", "missing")
PREV_RESULTS = MODEL_RESULTS + ("first inspection",)
TOP_FACILITY_TYPES = 20  # the 20 most common types cover 98% of modeling rows in the EDA

# Chicago's bounding box. Every coordinate in the EDA falls inside it.
LAT_BOUNDS = (41.64, 42.03)
LON_BOUNDS = (-87.94, -87.52)

# --- Features ---------------------------------------------------------------------------------

ZIP_RATE_WINDOW_DAYS = 365
ZIP_RATE_SMOOTHING = 50  # pseudo-inspections at the prior rate, so small ZIP codes stay near it
COMPLAINT_WINDOW_DAYS = 90

TARGET = "fail"
ID_COLUMNS = ["inspection_id", "license_number", "inspection_date"]

CATEGORICAL_FEATURES = ["inspection_group", "facility_group", "risk", "prev_result"]
NUMERIC_FEATURES = {
    "v1": ["prior_fails", "n_prior", "days_since_prev", "prev_n_violations"],
    "v2": ["prior_fails", "n_prior", "days_since_prev", "prev_n_violations",
           "zip_fail_rate_365d", "zip_complaints_90d"],
}
# v1 uses the establishment's own history. v2 adds two neighborhood features for its ZIP code.
FEATURE_COLUMNS = {version: CATEGORICAL_FEATURES + numeric for version, numeric in NUMERIC_FEATURES.items()}
FEATURE_VERSION = os.getenv("FEATURE_VERSION", "v1")  # FEATURE_VERSION
