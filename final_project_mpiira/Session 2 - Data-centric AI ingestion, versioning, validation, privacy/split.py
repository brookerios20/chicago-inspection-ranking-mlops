"""Session 2: split inspections by date into drift, train, validation and test sets.

Splitting by date (not at random) mirrors how the model is used: it learns from past inspections
and is judged on later ones. The test period stays untouched until the final check, as the project
brief requires. Inspections from before the July 2018 food code change form a separate "drift" set,
used in Part D to check that the monitoring notices when incoming data looks different.
"""
import pandas as pd

import config

SPLITS = ["drift", "train", "val", "test"]

def assign_split(df):
    """Label each inspection by period: before the 2018 code change, train, validation or test."""
    bins = [pd.Timestamp.min, config.CODE_CHANGE_DATE, config.VAL_START,
            config.TEST_START, pd.Timestamp.max]
    out = df.copy()
    out["split"] = pd.cut(out["inspection_date"], bins=bins, right=False, labels=SPLITS)
    return out

def split_file(clean_path, data_dir):
    """Read the cleaned parquet and write one parquet file per split."""
    df = assign_split(pd.read_parquet(clean_path))
    for name in SPLITS:
        df[df["split"] == name].to_parquet(data_dir / f"{name}.parquet", index=False)
    return df
