"""Stage 2: turn the raw snapshot into modeling rows with a target and cleaned categories.

The rules follow the EDA notebook (`eda/eda_rios.ipynb`):

- Modeling rows are Pass, Pass w/ Conditions, and Fail. The other results record visits that
  produced no inspection outcome.
- The target `fail` is 1 for Fail. The Fail rate stays stable across the 2018 food code change,
  while Fail or Pass w/ Conditions more than doubles after it.
- Rows with a missing or zero license cannot be linked to earlier inspections and are dropped.
- Inspection types collapse into five groups, and facility types keep the 20 most common
  values. The 20 are counted on rows before the validation window, so later data does not
  shape the categories.
"""

import re

import numpy as np
import pandas as pd

from src import config

RE_INSPECTION = re.compile(r"\bre[- ]?insp")
VIOLATION_CODE = re.compile(r"(?:^|\|)\s*(\d{1,2})\.")
RISK_MAP = {"Risk 1 (High)": "high", "Risk 2 (Medium)": "medium", "Risk 3 (Low)": "low"}

CLEAN_COLUMNS = [
    "inspection_id", "inspection_id_num", "license_number", "inspection_date", "results", "fail",
    "inspection_group", "is_canvass", "facility_group", "risk", "zip", "latitude", "longitude",
    "n_violations",
]


def inspection_group(text) -> str:
    """Collapse a lowercased inspection type into canvass, license, complaint, re-inspection, or other."""
    if pd.isna(text):
        return "other"
    if RE_INSPECTION.search(text):
        return "re-inspection"
    if "complaint" in text or "food poisoning" in text:
        return "complaint"
    if "canvass" in text:
        return "canvass"
    if "license" in text:
        return "license"
    return "other"


def count_violations(text) -> int:
    """Number of numbered violation codes cited in one inspection's violations field."""
    return len(VIOLATION_CODE.findall(text)) if isinstance(text, str) else 0


def clean(raw: pd.DataFrame, category_cutoff: pd.Timestamp = config.VALID_START) -> pd.DataFrame:
    """Filter to modeling rows and add the target and cleaned categories.

    Rows come back sorted by license, date, and inspection ID, the order history features need.
    """
    df = raw[raw["results"].isin(config.MODEL_RESULTS)].copy()

    license_number = df["license_number"].str.strip()
    linkable = license_number.notna() & ~license_number.isin(["0", "0.0"])
    df = df[linkable].copy()
    df["license_number"] = license_number[linkable].astype(str)

    df["results"] = df["results"].astype(str)
    df[config.TARGET] = (df["results"] == "Fail").astype("int8")
    df["inspection_id_num"] = pd.to_numeric(df["inspection_id"], errors="coerce")

    inspection_type = df["inspection_type"].str.lower().str.strip()
    df["inspection_group"] = inspection_type.map(inspection_group).astype(str)
    df["is_canvass"] = df["inspection_group"].eq("canvass")

    # Missing facility type is its own category, because those rows failed at 53% in the EDA.
    facility = df["facility_type"].str.lower().str.strip()
    top = facility[df["inspection_date"] < category_cutoff].value_counts().head(config.TOP_FACILITY_TYPES).index
    df["facility_group"] = facility.where(facility.isin(top), "other").mask(facility.isna(), "missing").astype(str)

    # "All" is not a risk level, so it is treated as missing along with empty values.
    df["risk"] = df["risk"].map(RISK_MAP).fillna("missing").astype(str)

    zip_code = df["zip"].str.strip().str[:5]
    valid_zip = zip_code.str.fullmatch(r"\d{5}").fillna(False).astype(bool)
    df["zip"] = zip_code.astype(object).where(valid_zip, None)

    df["n_violations"] = df["violations"].map(count_violations).astype(np.int64)

    df = df.sort_values(["license_number", "inspection_date", "inspection_id_num"], kind="mergesort")
    return df[CLEAN_COLUMNS].reset_index(drop=True)
