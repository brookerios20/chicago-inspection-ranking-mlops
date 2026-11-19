"""Ranking metrics shared by the baselines, model training, and monitoring.

The headline metric is precision@k on canvass inspections: the Fail rate among the k
highest-ranked inspections, with k a share of the ranked rows (`config.HEADLINE_K`). Canvass
inspections are the routine visits the city can reorder. Complaint, license, and
re-inspection visits happen whether or not a model ranks them.

Ties in the scores are broken by a seeded random draw, so a ranking with many equal scores
(such as a single-category baseline) scores like a random pick within the tie.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config

SLICE_COLUMNS = ["slice_risk", "slice_history", "slice_region", "slice_facility"]


def _top_k(scores: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Indices of the k highest scores, with ties broken at random."""
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)
    tiebreak = np.random.default_rng(seed).random(len(scores))
    return np.lexsort((tiebreak, -scores))[:k]


def precision_at_k(y_true, scores, frac: float = config.HEADLINE_K, seed: int = config.SEED) -> float:
    """Fail rate among the top `frac` share of rows ranked by score."""
    y = np.asarray(y_true, dtype=float)
    if len(y) == 0:
        return float("nan")
    k = max(1, int(frac * len(y)))
    return float(y[_top_k(scores, k, seed)].mean())


def standard_error(p: float, n: int) -> float:
    """Standard error of a proportion p measured on n rows."""
    return float(np.sqrt(p * (1 - p) / n)) if n else float("nan")


def k_name(frac: float) -> str:
    """Metric name for a k share, for example precision_at_10 for 0.10."""
    return f"precision_at_{round(frac * 100)}"


def ranking_metrics(y_true, scores, seed: int = config.SEED) -> dict:
    """Precision@k for each share in `config.K_FRACS`, lift, and threshold-free diagnostics.

    lift_per_100 is the number of extra failures found per 100 inspections at the headline k,
    compared with picking inspections at random.
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    base_rate = float(y.mean()) if len(y) else float("nan")
    out = {"n": len(y), "base_rate": base_rate}
    for frac in config.K_FRACS:
        out[k_name(frac)] = precision_at_k(y, s, frac, seed)
    headline = out[k_name(config.HEADLINE_K)]
    out["se_at_headline"] = standard_error(headline, max(1, int(config.HEADLINE_K * len(y))))
    out["lift_per_100"] = 100 * (headline - base_rate)
    has_both = len(np.unique(y)) == 2
    filled = np.nan_to_num(s, nan=np.nanmin(s) if np.isfinite(s).any() else 0.0)
    out["roc_auc"] = float(roc_auc_score(y, filled)) if has_both else float("nan")
    out["pr_auc"] = float(average_precision_score(y, filled)) if has_both else float("nan")
    return out


def add_slice_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the columns that define evaluation slices.

    - slice_risk: city risk category
    - slice_history: first inspection of the license, or a repeat
    - slice_region: north or south of Roosevelt Road, or unknown without coordinates
    - slice_facility: restaurant, grocery store, school, or other
    """
    lat = df["latitude"]
    return df.assign(
        slice_risk=df["risk"],
        slice_history=np.where(df["n_prior"].eq(0), "first inspection", "repeat"),
        slice_region=np.where(lat.isna(), "unknown", np.where(lat < config.ROOSEVELT_LAT, "south", "north")),
        slice_facility=df["facility_group"].where(df["facility_group"].isin(config.SLICE_FACILITIES), "other"),
    )


def slice_metrics(df: pd.DataFrame, score_col: str = "score", frac: float = config.HEADLINE_K,
                  seed: int = config.SEED) -> pd.DataFrame:
    """Precision@k and its standard error within each slice value."""
    df = add_slice_columns(df)
    rows = []
    for slice_col in SLICE_COLUMNS:
        for value, part in df.groupby(slice_col):
            p = precision_at_k(part[config.TARGET], part[score_col], frac, seed)
            k = max(1, int(frac * len(part)))
            rows.append({
                "slice": slice_col.removeprefix("slice_"),
                "value": value,
                "n": len(part),
                "base_rate": float(part[config.TARGET].mean()),
                k_name(frac): p,
                "se": standard_error(p, k),
            })
    return pd.DataFrame(rows)
