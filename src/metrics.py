"""Ranking metrics shared by the baselines, model training, and monitoring.

The headline metric is weekly precision@k on canvass inspections. Each week, the model ranks
that week's canvass inspections, and the top k are the visits the city would make first. The
metric is the Fail rate among all the visits picked across the window's weeks. k comes from
inspector capacity: `config.CAPACITY_SHARE` of the median number of canvass inspections per
week in the training window (`capacity_summary`). Canvass inspections are the routine visits
the city can reorder. Complaint, license, and re-inspection visits happen whether or not a
model ranks them.

Secondary metrics rank the whole window at once: precision@k at shares of the ranked rows
(`config.K_FRACS`), ROC-AUC, and PR-AUC. Slices use precision@k at `config.HEADLINE_K`,
because a slice has too few inspections in a single week.

Ties in the scores are broken by a seeded random draw, so a ranking with many equal scores
(such as a single-category baseline) scores like a random pick within the tie.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config

SLICE_COLUMNS = ["slice_risk", "slice_history", "slice_region", "slice_facility"]
HEADLINE = "weekly_precision_at_k"


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


def weeks(dates: pd.Series) -> pd.Series:
    """Calendar week of each inspection date."""
    return pd.to_datetime(dates).dt.to_period(config.WEEK_FREQ)


def capacity_summary(train: pd.DataFrame, share: float = config.CAPACITY_SHARE) -> dict:
    """Weekly inspection counts in the training window and the k they imply.

    k_week is `share` of the median number of canvass inspections per week, rounded, and at
    least 1. Weeks with no inspections are not counted.
    """
    counts = train.assign(week=weeks(train["inspection_date"])).groupby("week").agg(
        inspections=(config.TARGET, "size"), canvass=("is_canvass", "sum"))
    median_canvass = float(counts["canvass"].median())
    return {
        "weeks": len(counts),
        "median_weekly_inspections": float(counts["inspections"].median()),
        "median_weekly_canvass": median_canvass,
        "capacity_share": share,
        "k_week": max(1, int(round(share * median_canvass))),
    }


def weekly_precision_at_k(y_true, scores, dates, k: int, seed: int = config.SEED) -> tuple[float, int]:
    """Fail rate among the top k inspections of each week, pooled over the weeks.

    A week with fewer than k inspections contributes all of them. Returns the precision and the
    number of inspections picked.
    """
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "s": np.asarray(scores, dtype=float),
                          "week": weeks(pd.Series(dates)).to_numpy()})
    hits, picked = 0.0, 0
    for _, week in frame.groupby("week", sort=True):
        top = _top_k(week["s"].to_numpy(), min(k, len(week)), seed)
        hits += week["y"].to_numpy()[top].sum()
        picked += len(top)
    return (hits / picked if picked else float("nan")), picked


def ranking_metrics(y_true, scores, dates, k_week: int, seed: int = config.SEED) -> dict:
    """The headline weekly precision@k, the shares in `config.K_FRACS`, lift, and diagnostics.

    lift_per_100 is the number of extra failures found per 100 visits at the headline, compared
    with picking inspections at random.
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    base_rate = float(y.mean()) if len(y) else float("nan")
    headline, picked = weekly_precision_at_k(y, s, dates, k_week, seed)
    out = {"n": len(y), "base_rate": base_rate, "k_week": k_week, "picked": picked,
           HEADLINE: headline, "se_headline": standard_error(headline, picked),
           "lift_per_100": 100 * (headline - base_rate)}
    for frac in config.K_FRACS:
        out[k_name(frac)] = precision_at_k(y, s, frac, seed)
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
    """Precision@k and its standard error within each slice value with more than one value per slice."""
    df = add_slice_columns(df)
    rows = []
    for slice_col in SLICE_COLUMNS:
        if df[slice_col].nunique() < 2:
            continue  # one value (for example facility type under the restaurant scope)
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
