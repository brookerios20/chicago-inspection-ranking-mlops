"""Hashing and JSON helpers shared by the ingest and split stages.

A content hash identifies a table by its values, so it can confirm that two runs, or two
machines, hold the same data. Hashes are computed on a table read back from parquet, which is
the form every later stage loads.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd


def frame_hash(df: pd.DataFrame, sort_by: str = "inspection_id") -> str:
    """SHA-256 of a table's values, independent of row order when `sort_by` is a column."""
    if sort_by in df.columns:
        df = df.sort_values(sort_by, kind="mergesort")
    row_hashes = pd.util.hash_pandas_object(df.reset_index(drop=True), index=False)
    return hashlib.sha256(row_hashes.to_numpy().tobytes()).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes, read in chunks so large files fit in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_parquet(df: pd.DataFrame, path: Path) -> str:
    """Write `df` to parquet and return the content hash of the table as read back."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return frame_hash(pd.read_parquet(path))


def write_json(obj: dict, path: Path) -> None:
    """Write a manifest as indented JSON. Dates and other non-JSON values are written as text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def read_json(path: Path) -> dict:
    """Read a manifest written by `write_json`."""
    return json.loads(Path(path).read_text())
