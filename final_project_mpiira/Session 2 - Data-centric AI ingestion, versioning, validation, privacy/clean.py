"""Session 2: keep real inspection outcomes, build the fail label, tidy key columns."""
import pandas as pd

import config

RISK_LEVELS = {"Risk 1 (High)": 1, "Risk 2 (Medium)": 2, "Risk 3 (Low)": 3}

def clean(raw):
    """Return in-scope inspections with a real outcome, a 0/1 fail label and tidy columns."""
    df = raw[raw["results"].isin(config.FAIL_RESULTS + config.PASS_RESULTS)].copy()
    df = df[df["license_no"].notna() & (df["license_no"] != "0")]
    df["facility_type"] = df["facility_type"].str.strip().str.lower()
    df["inspection_type"] = df["inspection_type"].str.strip().str.lower()
    df = df[df["facility_type"].isin(config.FACILITY_TYPES)]
    df["risk_level"] = df["risk"].map(RISK_LEVELS)    # "All" and missing become NaN
    df["fail"] = df["results"].isin(config.FAIL_RESULTS).astype(int)
    return df.sort_values(["license_no", "inspection_date"]).reset_index(drop=True)

def clean_file(raw_path, clean_path):
    """Read the raw parquet, clean it and write the cleaned parquet."""
    df = clean(pd.read_parquet(raw_path))
    df.to_parquet(clean_path, index=False)
    return df
