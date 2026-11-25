"""Governance gates: a fairness gate on the weekly picks, and the privacy rule for logged requests.

The fairness gate turns the Session 6 Fairlearn check into a pass/fail rule. It ranks each week's
inspections, takes the top k as that week's visits, and compares two groups (for example the
North and South Sides) on two measures:

- selection rate: the share of each group's inspections that are picked (demographic parity)
- precision of the picks: the fail rate among each group's picked inspections

The gate passes when both gaps are at most `MAX_GAP`. The limit is a policy choice, set here at
10 percentage points; the production model's gaps on the validation year are about 3 and 2 points.

The privacy rule lists the only fields a request log may hold: the request ID and the model's
features, none of which names or locates a business.
"""

import pandas as pd

from src import config, metrics

MAX_GAP = 0.10   # largest allowed gap between groups, in selection rate and in precision of the picks

# Fields a logged request may contain: the client's ID for the inspection and the model's features.
LOGGABLE_FIELDS = {"facility_id", *config.FEATURE_COLUMNS["v1"], *config.FEATURE_COLUMNS["v2"]}


def weekly_picks(frame: pd.DataFrame, score_col: str, k: int) -> pd.Series:
    """True for the k highest-scoring inspections of each week, the visits the city would make."""
    week = metrics.weeks(frame["inspection_date"])
    return frame.groupby(week)[score_col].rank(method="first", ascending=False) <= k


def fairness_gate(frame: pd.DataFrame, group_col: str, k: int, score_col: str = "prediction",
                  max_gap: float = MAX_GAP) -> tuple[bool, pd.DataFrame, dict]:
    """Compare the groups in `group_col` on the weekly top-k picks.

    Returns whether the gate passed, a table per group (inspections, fail rate, selection rate,
    precision of the picks), and the gap between the groups for each measure.
    """
    picked = frame.assign(picked=weekly_picks(frame, score_col, k))
    table = picked.groupby(group_col).apply(lambda group: pd.Series({
        "inspections": len(group),
        "fail rate": group[config.TARGET].mean(),
        "selection rate": group["picked"].mean(),
        "precision of picks": group.loc[group["picked"], config.TARGET].mean(),
    }), include_groups=False)
    gaps = {measure: float(table[measure].max() - table[measure].min())
            for measure in ["selection rate", "precision of picks"]}
    return all(gap <= max_gap for gap in gaps.values()), table, gaps


def unloggable_fields(logged_fields) -> set[str]:
    """Fields in a request log that the privacy rule does not allow, such as names or addresses."""
    return set(logged_fields) - LOGGABLE_FIELDS
