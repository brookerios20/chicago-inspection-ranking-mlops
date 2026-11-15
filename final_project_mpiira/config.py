"""Project-wide constants shared by the notebook, the Airflow DAG and the API."""
from pathlib import Path

import pandas as pd

# Paths (built from this file's location, so they work from any working directory)
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Data"
RAW_CSV = DATA_DIR / "Food_Inspections_20261115.csv"
RAW_PARQUET = DATA_DIR / "inspections_raw.parquet"
CLEAN_PARQUET = DATA_DIR / "inspections_clean.parquet"

# One folder per course session; each holds that session's modules and assets
SESSION_DIRS = {
    "1": PROJECT_DIR / "Session 1 - DevOps to MLOps, CI-CD, basic serving",
    "2": PROJECT_DIR / "Session 2 - Data-centric AI ingestion, versioning, validation, privacy",
    "3": PROJECT_DIR / "Session 3 - Data engineering and ML pipelines",
    "4": PROJECT_DIR / "Session 4 - AutoML, training pipelines and orchestration",
    "5": PROJECT_DIR / "Session 5 - Deployment engineering",
    "6": PROJECT_DIR / "Session 6 - Explainability, fairness, faster inference",
    "7a": PROJECT_DIR / "Session 7a - Testing and deployment testing",
    "7b": PROJECT_DIR / "Session 7b - Monitoring and drift simulation",
    "8": PROJECT_DIR / "Session 8 - Ethics and governance gates",
    "9": PROJECT_DIR / "Session 9 - MLOps in practice",
}

# Reproducibility
SEED = 42

# Label definition: outcome of an inspection
FAIL_RESULTS = ["Fail"]
PASS_RESULTS = ["Pass", "Pass w/ Conditions"]
FACILITY_TYPES = ["restaurant"]    

# Key dates
CODE_CHANGE_DATE = pd.Timestamp("2018-07-01")    # food code overhaul, new violation codes
VAL_START = pd.Timestamp("2024-11-01")           # start of the validation period
TEST_START = pd.Timestamp("2025-11-01")          # start of the held-out test period
