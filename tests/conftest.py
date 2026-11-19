"""Shared fixtures: small hand-built inspection tables in the raw export's format."""

import pandas as pd
import pytest

from src.ingest import coerce_types


def make_raw(rows: list[dict]) -> pd.DataFrame:
    """Build a raw snapshot from partial rows, filling the other columns with plausible defaults.

    Each row needs inspection_id, license_number, inspection_date (MM/DD/YYYY), and results.
    The frame passes through `coerce_types`, so its types match the output of ingest.
    """
    defaults = {
        "dba_name": "TEST KITCHEN", "aka_name": "TEST KITCHEN", "facility_type": "Restaurant",
        "risk": "Risk 1 (High)", "address": "1 N STATE ST", "city": "CHICAGO", "state": "IL",
        "zip": "60601", "inspection_type": "Canvass",
        "violations": "1. FIRST VIOLATION | 2. SECOND VIOLATION",
        "latitude": "41.88", "longitude": "-87.63",
    }
    return coerce_types(pd.DataFrame([{**defaults, **row} for row in rows]))


@pytest.fixture
def same_day_raw() -> pd.DataFrame:
    """One license with a Fail and a same-day re-inspection Pass, then a canvass visit 56 days later."""
    return make_raw([
        {"inspection_id": "1", "license_number": "100", "inspection_date": "01/06/2020", "results": "Fail"},
        {"inspection_id": "2", "license_number": "100", "inspection_date": "01/06/2020", "results": "Pass",
         "inspection_type": "Canvass Re-Inspection"},
        {"inspection_id": "3", "license_number": "100", "inspection_date": "03/02/2020", "results": "Pass",
         "violations": None},
    ])


@pytest.fixture
def split_raw() -> pd.DataFrame:
    """Two inspections in each split window, from two licenses."""
    dates = ["06/29/2018", "06/30/2018", "07/01/2018", "10/29/2024",
             "10/30/2024", "10/29/2025", "10/30/2025", "10/29/2026"]
    return make_raw([
        {"inspection_id": str(i), "license_number": str(100 + i % 2), "inspection_date": d,
         "results": "Fail" if i % 3 == 0 else "Pass"}
        for i, d in enumerate(dates, start=1)
    ])
