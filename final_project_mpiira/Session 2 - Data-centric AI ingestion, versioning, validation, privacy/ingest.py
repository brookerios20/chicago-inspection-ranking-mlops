"""Session 2: load the raw Food Inspections CSV and save a typed parquet snapshot."""
import pandas as pd

def load_raw(csv_path):
    """Read the raw CSV with stable column types and snake_case column names."""
    df = pd.read_csv(csv_path, dtype={"License #": "string", "Zip": "string"})
    df.columns = df.columns.str.strip().str.lower()
    df.columns = df.columns.str.replace("#", "no").str.replace(" ", "_")
    df["inspection_date"] = pd.to_datetime(df["inspection_date"], format="%m/%d/%Y")
    return df

def ingest(csv_path, parquet_path):
    """Load the raw CSV and write it to parquet for the next pipeline steps."""
    df = load_raw(csv_path)
    df.to_parquet(parquet_path, index=False)
    return df
