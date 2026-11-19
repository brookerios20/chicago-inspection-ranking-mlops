"""Tests for the fixed-date splits and the test-set guard."""

import pandas as pd
import pytest

from src.clean import clean
from src.features import build_features
from src.manifest import read_json
from src.split import HeldOutSplitError, assign_split, load_split, write_splits


@pytest.fixture
def split_dir(split_raw, tmp_path):
    write_splits(build_features(clean(split_raw)), out_dir=tmp_path)
    return tmp_path


def test_boundaries():
    dates = pd.to_datetime(pd.Series([
        "2018-06-30", "2018-07-01", "2024-10-29", "2024-10-30", "2025-10-29", "2025-10-30", "2026-10-29",
        "2026-10-30",
    ]))
    assert list(assign_split(dates)) == [
        "drift_pre2018", "train", "train", "valid", "valid", "test", "test", "after_snapshot",
    ]


def test_splits_are_ordered_and_disjoint(split_dir):
    splits = read_json(split_dir / "split_manifest.json")["splits"]
    order = ["drift_pre2018", "train", "valid", "test"]
    for earlier, later in zip(order, order[1:]):
        assert splits[earlier]["date_max"] < splits[later]["date_min"]
    assert sum(s["rows"] for s in splits.values()) == 8


def test_test_split_requires_permission(split_dir):
    with pytest.raises(HeldOutSplitError):
        load_split("test", data_dir=split_dir)
    assert len(load_split("test", allow_test=True, data_dir=split_dir)) == 2


def test_changed_split_fails_hash_check(split_dir):
    train = pd.read_parquet(split_dir / "train.parquet")
    train.loc[0, "fail"] = 1 - train.loc[0, "fail"]
    train.to_parquet(split_dir / "train.parquet", index=False)
    with pytest.raises(ValueError, match="manifest hash"):
        load_split("train", data_dir=split_dir)


def test_rows_after_snapshot_are_rejected(split_raw, tmp_path):
    features = build_features(clean(split_raw))
    features.loc[0, "inspection_date"] = pd.Timestamp("2027-01-04")
    with pytest.raises(ValueError, match="after the snapshot end"):
        write_splits(features, out_dir=tmp_path)
