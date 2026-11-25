"""Production monitoring for the prediction API: request logging, drift reports, alerts, drift scenarios.

The monitor checks served traffic in three layers, from the fastest signal to the slowest:

1. Request checks, on every batch: the share of requests the API rejected (HTTP 422), and the
   rules in `validate.validate_features` that flag values outside plausible ranges or unknown
   categories in the requests it accepted.
2. Drift checks, on a monitoring window: Evidently compares the served inputs and the returned
   scores with the reference data (the validation canvass inspections), column by column.
3. Performance checks, once inspection results arrive: weekly precision@k of the served scores.
   Results arrive only after the visits, so this layer lags the other two.

Every batch sent to the API is written to a JSON lines log (one line per batch, rejected batches
included), and the checks read the log, which is what the API actually received and returned.

The functions at the end corrupt a clean batch in the ways the drift simulation tests: impossible
values, swapped columns, a renamed field, a changed category format and shuffled outcomes.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import bar_plot_panel
from evidently.ui.workspace import Workspace

from src import config, metrics, validate

MONITORING_DIR = config.PROJECT_ROOT / "monitoring"   # request log, alert log, Evidently workspace, reports
DEFAULT_THRESHOLD = 0.1     # Evidently's default for its distance measures (Wasserstein, Jensen-Shannon)
THRESHOLD_MARGIN = 1.5      # a column alerts when its drift exceeds 1.5 times the drift seen on clean data
DRIFT_SHARE_LIMIT = 0.2     # dataset alert when more than 20% of the monitored columns drift
PRECISION_SE_LIMIT = 2      # performance alert when precision@k falls 2 standard errors below validation

logger = logging.getLogger("inspectrank.monitoring")


def monitored_columns(version: str) -> list[str]:
    """The model's input columns for feature `version`, plus the score the API returned."""
    return list(config.FEATURE_COLUMNS[version]) + ["score"]


def data_definition(version: str) -> DataDefinition:
    """The column types Evidently needs: numeric features and the score, and categorical features."""
    return DataDefinition(numerical_columns=list(config.NUMERIC_FEATURES[version]) + ["score"],
                          categorical_columns=list(config.CATEGORICAL_FEATURES))


# ---- request log -------------------------------------------------------------------------------

def to_requests(rows: pd.DataFrame, columns: list[str]) -> list[dict]:
    """Rows as API facility records, with the inspection ID as facility_id and missing values as null."""
    records = json.loads(rows[columns].to_json(orient="records"))
    return [{"facility_id": str(inspection_id), **record} for inspection_id, record in zip(rows["inspection_id"], records)]


def send_weekly(rows: pd.DataFrame, url: str, scenario: str, log_path: Path, columns: list[str]) -> pd.DataFrame:
    """POST each week's rows to the API as one batch, log every batch, and return the rows with their scores.

    Each log line holds the time, the scenario, the week, the request, the HTTP status and the
    response. A rejected batch is logged with its error, and its rows get no score.
    """
    scored = []
    with open(log_path, "a") as log:
        for week, batch in rows.groupby(metrics.weeks(rows["inspection_date"]), sort=True):
            facilities = to_requests(batch, columns)
            response = requests.post(f"{url}/predict", json={"facilities": facilities}, timeout=60)
            log.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "scenario": scenario,
                                  "week": str(week), "status": response.status_code,
                                  "request": facilities, "response": response.json()}) + "\n")
            scores = {r["facility_id"]: r["score"] for r in response.json().get("results", [])}
            scored.append(batch.assign(status=response.status_code,
                                       score=batch["inspection_id"].astype(str).map(scores)))
    return pd.concat(scored)


def read_log(log_path: Path) -> pd.DataFrame:
    """One row per logged facility: scenario, week, HTTP status, the request's fields and the returned score."""
    rows = []
    for line in Path(log_path).read_text().splitlines():
        batch = json.loads(line)
        scores = {r["facility_id"]: r["score"] for r in batch["response"].get("results", [])}
        rows += [{"scenario": batch["scenario"], "week": batch["week"], "status": batch["status"], **facility,
                  "score": scores.get(facility["facility_id"])} for facility in batch["request"]]
    return pd.DataFrame(rows)


# ---- drift -------------------------------------------------------------------------------------

def drift_report(current: pd.DataFrame, reference: pd.DataFrame, version: str, thresholds: dict | None = None,
                 name: str | None = None, timestamp: datetime | None = None):
    """Evidently drift report of the current rows against the reference, with a pass/fail test per column.

    Numeric columns use the normed Wasserstein distance and categorical columns the Jensen-Shannon
    distance (Evidently's defaults for samples over 1,000 rows). `thresholds` sets each column's
    limit; without it every column uses Evidently's default of 0.1.
    """
    columns = monitored_columns(version)
    preset = DataDriftPreset(columns=columns, drift_share=DRIFT_SHARE_LIMIT, per_column_threshold=thresholds)
    definition = data_definition(version)
    return Report([preset], include_tests=True).run(
        Dataset.from_pandas(current[columns], data_definition=definition),
        Dataset.from_pandas(reference[columns], data_definition=definition),
        name=name, timestamp=timestamp)


def drift_scores(snapshot) -> pd.Series:
    """Each monitored column's drift score, and the share of drifted columns, from an Evidently report."""
    values = {}
    for metric in snapshot.dict()["metrics"]:
        value = metric["value"]
        values[metric["config"].get("column", "drifted_share")] = value["share"] if isinstance(value, dict) else value
    return pd.Series(values)


def calibrate(clean_scores: pd.Series) -> dict:
    """Per-column drift thresholds: the drift measured on clean data times the margin, at least the default."""
    return {column: round(max(DEFAULT_THRESHOLD, THRESHOLD_MARGIN * score), 3)
            for column, score in clean_scores.drop("drifted_share").items()}


def create_dashboard(workspace_path: Path, version: str, name: str = "InspectRank monitoring"):
    """A local Evidently workspace with one project, whose dashboard plots the drift checks of every run.

    The panels show, per monitoring run, the share of drifted columns (the dataset alert), the
    drift of the returned scores, and the drift of each numeric feature. `evidently ui` serves it.
    """
    workspace = Workspace.create(str(workspace_path))
    project = workspace.create_project(name)
    project.description = "Served requests and scores of the inspection-ranker API against the validation reference"
    project.save()
    project.dashboard.add_panel(bar_plot_panel(
        "Share of drifted columns", size="half", description=f"dataset alert above {DRIFT_SHARE_LIMIT}",
        values=[PanelMetric(legend="share", metric="DriftedColumnsCount", metric_labels={"value_type": "share"})]))
    project.dashboard.add_panel(bar_plot_panel(
        "Drift of the returned scores", size="half", description="distance from the reference scores",
        values=[PanelMetric(legend="score", metric="ValueDrift", metric_labels={"column": "score"})]))
    project.dashboard.add_panel(bar_plot_panel(
        "Drift of each numeric feature", description="distance from the reference distribution",
        values=[PanelMetric(legend=column, metric="ValueDrift", metric_labels={"column": column})
                for column in config.NUMERIC_FEATURES[version]]))
    return workspace, project


# ---- alerts ------------------------------------------------------------------------------------

def check(scenario_log: pd.DataFrame, reference: pd.DataFrame, version: str, thresholds: dict,
          labels: pd.DataFrame, k: int, valid_precision: float, valid_se: float, name: str,
          timestamp: datetime):
    """Run the three layers of checks on one scenario's logged traffic.

    Returns a table with one row per check (value, limit, whether it alerts, detail) and the
    Evidently report, which is None when the API rejected every request. `labels` holds the
    inspection results (inspection_id, inspection_date, fail) that arrive after the visits.
    """
    accepted = scenario_log[scenario_log["status"] == 200]
    rules = validate.validate_features(accepted.drop(columns=["scenario", "week", "status", "facility_id", "score"]),
                                       version)
    warnings = validate.warnings_of(rules)
    rejected = 1 - len(accepted) / len(scenario_log)
    rows = [
        {"check": "rejected requests (share)", "value": rejected, "limit": 0.0, "alert": rejected > 0,
         "detail": f"{len(scenario_log) - len(accepted)} of {len(scenario_log)} rejected"},
        {"check": "out-of-range or unknown values (rules)", "value": float(len(warnings)), "limit": 0.0,
         "alert": len(warnings) > 0,
         "detail": "; ".join(f"{r.rule}: {r.detail}" for r in warnings.itertuples()) or "none"},
    ]
    snapshot = None
    if len(accepted):
        snapshot = drift_report(accepted, reference, version, thresholds, name=name, timestamp=timestamp)
        scores = drift_scores(snapshot)
        drifted = [column for column in thresholds if scores[column] >= thresholds[column]]
        served = accepted.astype({"facility_id": str}).merge(
            labels.astype({"inspection_id": str}), left_on="facility_id", right_on="inspection_id")
        precision, _ = metrics.weekly_precision_at_k(served["fail"], served["score"], served["inspection_date"], k)
        precision_floor = valid_precision - PRECISION_SE_LIMIT * valid_se
        rows += [
            {"check": "drifted columns (share)", "value": scores["drifted_share"], "limit": DRIFT_SHARE_LIMIT,
             "alert": scores["drifted_share"] > DRIFT_SHARE_LIMIT, "detail": ", ".join(drifted) or "none"},
            {"check": "score drift (distance)", "value": scores["score"], "limit": thresholds["score"],
             "alert": scores["score"] >= thresholds["score"],
             "detail": f"mean score {accepted['score'].mean():.3f} (reference {reference['score'].mean():.3f})"},
            {"check": "weekly precision@k (after results arrive)", "value": precision, "limit": precision_floor,
             "alert": precision < precision_floor,
             "detail": f"validation {valid_precision:.3f}; limit is {PRECISION_SE_LIMIT} standard errors below it"},
        ]
    table = pd.DataFrame(rows)
    return table.assign(scenario=name), snapshot


def raise_alerts(table: pd.DataFrame, alert_log: Path) -> pd.DataFrame:
    """Write every check that alerts to the alert log and to the logging system, and return those rows."""
    alerts = table[table["alert"]]
    with open(alert_log, "a") as log:
        for row in alerts.itertuples():
            message = f"ALERT [{row.scenario}] {row.check}: {row.value:.3f} (limit {row.limit:.3f}); {row.detail}"
            log.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")
            logger.warning(message)
    return alerts


# ---- drift scenarios ---------------------------------------------------------------------------

def as_float(rows: pd.DataFrame, version: str) -> pd.DataFrame:
    """A copy of the rows with every numeric feature as float, so corrupted values fit in the column."""
    return rows.astype({column: float for column in config.NUMERIC_FEATURES[version]})


def out_of_bounds(rows: pd.DataFrame, version: str, share: float = 0.2, seed: int = config.SEED) -> pd.DataFrame:
    """Impossible random values in a share of rows: 250 to 1,000 earlier fails, ZIP fail rates of 1.5 to 5."""
    rng = np.random.default_rng(seed)
    rows = as_float(rows, version)
    hit = rng.random(len(rows)) < share
    rows.loc[hit, "prior_fails"] = rng.integers(250, 1000, hit.sum()).astype(float)
    rows.loc[hit, "zip_fail_rate_365d"] = rng.uniform(1.5, 5, hit.sum())
    return rows


def swap_columns(rows: pd.DataFrame, version: str, first: str = "prior_fails",
                 second: str = "n_prior") -> pd.DataFrame:
    """Two numeric columns swapped, as a client that maps fields by position would send them."""
    rows = as_float(rows, version)
    rows[[first, second]] = rows[[second, first]].to_numpy()
    return rows


def rename_field(rows: pd.DataFrame, old: str = "prior_fails", new: str = "prior_failures") -> pd.DataFrame:
    """A field renamed by the upstream system, so requests no longer match the API's schema."""
    return rows.rename(columns={old: new})


def raw_risk_labels(rows: pd.DataFrame) -> pd.DataFrame:
    """Risk levels in the city portal's raw format ("Risk 1 (High)") instead of the cleaned values."""
    raw = {"high": "Risk 1 (High)", "medium": "Risk 2 (Medium)", "low": "Risk 3 (Low)", "missing": "All"}
    return rows.assign(risk=rows["risk"].map(raw))


def shuffle_outcomes(rows: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """Outcomes shuffled within each week: the inputs are unchanged, but they no longer predict the outcome."""
    rng = np.random.default_rng(seed)
    week = metrics.weeks(rows["inspection_date"])
    return rows.assign(fail=rows.groupby(week)["fail"].transform(lambda fails: rng.permutation(fails.to_numpy())))
