"""Stage 4: divide the feature table into fixed-date splits and guard the test set.

The splits use fixed dates from `config`, so a newer export cannot move the boundaries:

| Split         | Inspection dates                       | Use                                   |
|---------------|----------------------------------------|---------------------------------------|
| drift_pre2018 | before 2018-07-01                      | drift demonstration (old food code)   |
| train         | 2018-07-01 to the day before VALID_START | model fitting                       |
| valid         | VALID_START to the day before TEST_START | baselines and model selection       |
| test          | TEST_START through SNAPSHOT_END        | production validation only            |

Each split is written to `data/processed/<name>.parquet`, and the split manifest records its
size, fail rates, date range, and content hash. `load_split` refuses the test split unless the
caller passes `allow_test=True`, which only the production validation script does.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.manifest import frame_hash, read_json, write_json, write_parquet

logger = logging.getLogger(__name__)


class HeldOutSplitError(PermissionError):
    """Raised when code other than production validation asks for the test split."""


def assign_split(dates: pd.Series) -> np.ndarray:
    """Split name for each inspection date."""
    return np.select(
        [dates < config.FOOD_CODE_CHANGE, dates < config.VALID_START, dates < config.TEST_START,
         dates <= config.SNAPSHOT_END],
        ["drift_pre2018", "train", "valid", "test"],
        default="after_snapshot",
    )


def _summary(part: pd.DataFrame) -> dict:
    """Size, fail rates, and date range of one split."""
    canvass = part[part["is_canvass"]]
    return {
        "rows": len(part),
        "canvass_rows": len(canvass),
        "fail_rate": round(float(part[config.TARGET].mean()), 4) if len(part) else None,
        "canvass_fail_rate": round(float(canvass[config.TARGET].mean()), 4) if len(canvass) else None,
        "date_min": part["inspection_date"].min().date() if len(part) else None,
        "date_max": part["inspection_date"].max().date() if len(part) else None,
    }


def write_splits(features: pd.DataFrame, out_dir: Path = config.PROCESSED_DIR,
                 raw_manifest: dict | None = None) -> dict:
    """Write each split to parquet with its manifest, and return the manifest."""
    names = assign_split(features["inspection_date"])
    late = int((names == "after_snapshot").sum())
    if late:
        raise ValueError(f"{late} rows fall after the snapshot end {config.SNAPSHOT_END.date()}.")

    entries = {}
    for name in config.SPLITS:
        part = features[names == name].reset_index(drop=True)
        path = Path(out_dir) / f"{name}.parquet"
        entries[name] = {"file": path.name, **_summary(part), "content_sha256": write_parquet(part, path)}
        logger.info("%s: %s rows (%s canvass)", name, f"{len(part):,}", f"{entries[name]['canvass_rows']:,}")

    manifest = {
        "facility_scope": config.FACILITY_SCOPE,
        "split_dates": {
            "food_code_change": config.FOOD_CODE_CHANGE.date(),
            "valid_start": config.VALID_START.date(),
            "test_start": config.TEST_START.date(),
            "snapshot_end": config.SNAPSHOT_END.date(),
        },
        "raw_content_sha256": (raw_manifest or {}).get("content_sha256"),
        "feature_columns": config.FEATURE_COLUMNS,
        "splits": entries,
    }
    write_json(manifest, Path(out_dir) / config.SPLIT_MANIFEST_NAME)
    return manifest


def load_split(name: str, allow_test: bool = False, data_dir: Path = config.PROCESSED_DIR,
               verify: bool = True) -> pd.DataFrame:
    """Load one split, checking it against the manifest hash.

    The test split stays held out until production validation, so loading it requires
    `allow_test=True`. Training and model selection never pass it.
    """
    if name not in config.SPLITS:
        raise ValueError(f"Unknown split {name!r}; expected one of {config.SPLITS}.")
    if name == "test" and not allow_test:
        raise HeldOutSplitError(
            "The test split is held out until production validation. "
            "Only the production validation script passes allow_test=True."
        )
    data_dir = Path(data_dir)
    df = pd.read_parquet(data_dir / f"{name}.parquet")
    if verify:
        expected = read_json(data_dir / config.SPLIT_MANIFEST_NAME)["splits"][name]["content_sha256"]
        if frame_hash(df) != expected:
            raise ValueError(f"Split {name!r} does not match its manifest hash. Rerun run_data.py.")
    return df
