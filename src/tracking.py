"""MLflow setup and the lineage tags that tie every run to its code and data version.

The pipeline logs through `MlflowClient` with explicit run IDs, so Prefect tasks can log to the
right run whichever thread runs them. A model is logged inside `active_run(run_id)`, because
MLflow's `log_model` needs an active run.

Lineage tags recorded on every run:

- git_commit and git_dirty: the code version, and whether uncommitted changes were present
- data_tag: the nearest `data-v*` git tag, the data version name
- data_processed_md5: the hash DVC records in dvc.lock for data/processed, the exact splits
- facility_scope and snapshot_end: the settings that shaped the data
"""

import json
import math
import subprocess
from contextlib import contextmanager
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient

from src import config


def client() -> MlflowClient:
    """A client for the configured tracking store."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    return MlflowClient(config.MLFLOW_TRACKING_URI)


def experiment_id(name: str = config.MLFLOW_EXPERIMENT) -> str:
    """ID of the project's experiment, created on first use with artifacts in mlartifacts/."""
    c = client()
    experiment = c.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    config.MLFLOW_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return c.create_experiment(name, artifact_location=config.MLFLOW_ARTIFACT_DIR.as_uri())


def _git(*args: str) -> str:
    """Output of a git command in the project root, or an empty string if it fails."""
    result = subprocess.run(["git", *args], cwd=config.PROJECT_ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def dvc_out_md5(path: str = "data/processed", lock: Path = config.PROJECT_ROOT / "dvc.lock") -> str:
    """The hash dvc.lock records for a stage output, or an empty string if it is not there."""
    if not Path(lock).exists():
        return ""
    stages = yaml.safe_load(Path(lock).read_text()).get("stages", {})
    for stage in stages.values():
        for out in stage.get("outs", []):
            if out.get("path") == path:
                return out.get("md5", "")
    return ""


def lineage_tags() -> dict:
    """Code and data version tags for a run."""
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": str(bool(_git("status", "--porcelain", "--untracked-files=no"))),
        "data_tag": _git("describe", "--tags", "--match", "data-*", "--abbrev=0"),
        "data_processed_md5": dvc_out_md5("data/processed"),
        "facility_scope": config.FACILITY_SCOPE,
        "snapshot_end": str(config.SNAPSHOT_END.date()),
    }


def start_run(name: str, parent_id: str | None = None, tags: dict | None = None) -> str:
    """Create a run (a child run when `parent_id` is given) and return its ID."""
    all_tags = {"mlflow.runName": name, **(tags or {})}
    if parent_id:
        all_tags["mlflow.parentRunId"] = parent_id
    return client().create_run(experiment_id(), tags=all_tags).info.run_id


def end_run(run_id: str, status: str = "FINISHED") -> None:
    """Mark a run finished (or FAILED)."""
    client().set_terminated(run_id, status=status)


def log_params(run_id: str, params: dict) -> None:
    """Log parameters, as text cut to MLflow's 6,000-character limit."""
    c = client()
    for key, value in params.items():
        c.log_param(run_id, key, str(value)[:6000])


def log_metrics(run_id: str, metrics: dict) -> None:
    """Log the numeric, finite values of `metrics`."""
    c = client()
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            c.log_metric(run_id, key, float(value))


def set_tags(run_id: str, tags: dict) -> None:
    """Set tags on a run."""
    c = client()
    for key, value in tags.items():
        c.set_tag(run_id, key, str(value))


def log_artifact(run_id: str, path: Path, artifact_path: str | None = None) -> None:
    """Upload a file to the run's artifacts."""
    client().log_artifact(run_id, str(path), artifact_path)


def log_dict(run_id: str, obj: dict, filename: str) -> None:
    """Store a dictionary as a JSON artifact."""
    client().log_text(run_id, json.dumps(obj, indent=2, default=str), filename)


@contextmanager
def active_run(run_id: str):
    """Make `run_id` the active MLflow run, for calls such as `mlflow.sklearn.log_model`."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=run_id) as run:
        yield run
