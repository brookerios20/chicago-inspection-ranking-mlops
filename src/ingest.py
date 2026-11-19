"""Stage 1: load the Food Inspections snapshot and record its manifest.

The stage reads the local CSV export when it exists and otherwise downloads the same rows
from the Chicago Data Portal API. Both sources are mapped to one snake_case schema, cut at
`config.SNAPSHOT_END`, and written to `data/raw/inspections.parquet`. The manifest beside it
records the source, the row count, the date range, and a content hash, so a rerun on another
machine can confirm it holds the same snapshot.
"""

import io
import logging
from datetime import datetime, timezone

import pandas as pd
import requests

from src import config
from src.manifest import file_sha256, frame_hash, read_json, write_json, write_parquet

logger = logging.getLogger(__name__)

NON_STRING_COLUMNS = {"inspection_date", "latitude", "longitude"}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map CSV headers ('License #', 'Inspection Date') and API names ('license_') to snake_case."""
    renamed = {}
    for col in df.columns:
        name = col.strip().lower().replace(" #", "_number").replace(" ", "_")
        if name in ("license_", "license"):
            name = "license_number"
        renamed[col] = name
    return df.rename(columns=renamed)


def read_local_csv(path=config.RAW_CSV) -> pd.DataFrame:
    """Read the portal's CSV export with every column as text, so IDs and ZIP codes keep their form."""
    return normalize_columns(pd.read_csv(path, dtype=str))


def download_api(end: pd.Timestamp = config.SNAPSHOT_END, page_size: int = config.API_PAGE_SIZE,
                 timeout: int = 120) -> pd.DataFrame:
    """Download every inspection up to `end` from the portal API, one page of rows at a time.

    Ordering by inspection ID keeps the pages stable, so no row is skipped or repeated.
    """
    where = f"inspection_date <= '{end:%Y-%m-%d}T23:59:59'"
    pages, offset = [], 0
    while True:
        params = {"$where": where, "$order": "inspection_id", "$limit": page_size, "$offset": offset}
        response = requests.get(config.API_URL, params=params, timeout=timeout)
        response.raise_for_status()
        page = pd.read_csv(io.StringIO(response.text), dtype=str)
        if page.empty:
            break
        pages.append(page)
        offset += len(page)
        logger.info("Downloaded %s rows", f"{offset:,}")
        if len(page) < page_size:
            break
    if not pages:
        raise ValueError("The portal API returned no rows.")
    return normalize_columns(pd.concat(pages, ignore_index=True))


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse the CSV format (MM/DD/YYYY) and the API format (ISO 8601) into dates."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return values.dt.normalize()
    text = values.astype(object)
    parsed = pd.to_datetime(text, format="%m/%d/%Y", errors="coerce")
    iso = pd.to_datetime(text.where(parsed.isna()), format="ISO8601", errors="coerce")
    return parsed.fillna(iso).dt.normalize()


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the schema columns, strip text, turn empty strings into missing, and parse types.

    Columns missing from the source are left out here and reported by validation.
    """
    df = df[[c for c in config.RAW_COLUMNS if c in df.columns]].copy()
    for col in df.columns:
        if col in NON_STRING_COLUMNS:
            continue
        text = df[col].astype("string").str.strip()
        df[col] = text.mask((text == "").fillna(False))
    if "inspection_date" in df.columns:
        df["inspection_date"] = parse_dates(df["inspection_date"])
    for col in ("latitude", "longitude"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def ingest(source: str = "auto") -> tuple[pd.DataFrame, dict]:
    """Load the snapshot, write the raw parquet and manifest, and return both.

    `source` is "local" (the CSV in `data/`), "api" (download), or "auto" (local when the
    CSV exists, otherwise the API).
    """
    use_local = source == "local" or (source == "auto" and config.RAW_CSV.exists())
    if use_local:
        raw = read_local_csv(config.RAW_CSV)
        origin = {"type": "local_csv", "file": config.RAW_CSV.name, "file_sha256": file_sha256(config.RAW_CSV)}
    elif source in ("api", "auto"):
        raw = download_api()
        origin = {"type": "api", "url": config.API_URL}
    else:
        raise ValueError(f"Unknown source {source!r}; use 'auto', 'local', or 'api'.")

    df = coerce_types(raw)
    # Rows with an unreadable date are kept so validation can report them.
    in_snapshot = df["inspection_date"].isna() | (df["inspection_date"] <= config.SNAPSHOT_END)
    df = df[in_snapshot].sort_values("inspection_id", kind="mergesort").reset_index(drop=True)

    content_hash = write_parquet(df, config.RAW_PARQUET)
    manifest = {
        "dataset_id": config.DATASET_ID,
        "source": origin,
        "snapshot_end": config.SNAPSHOT_END.date(),
        "rows": len(df),
        "date_min": df["inspection_date"].min().date(),
        "date_max": df["inspection_date"].max().date(),
        "columns": list(df.columns),
        "content_sha256": content_hash,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(manifest, config.RAW_MANIFEST)
    logger.info("Raw snapshot: %s rows, %s to %s", f"{len(df):,}", manifest["date_min"], manifest["date_max"])
    return df, manifest


def load_snapshot(verify: bool = True) -> pd.DataFrame:
    """Load the raw snapshot written by `ingest`, checking it against the manifest hash."""
    df = pd.read_parquet(config.RAW_PARQUET)
    if verify:
        expected = read_json(config.RAW_MANIFEST)["content_sha256"]
        if frame_hash(df) != expected:
            raise ValueError("The raw snapshot does not match its manifest hash. Rerun ingest.")
    return df
