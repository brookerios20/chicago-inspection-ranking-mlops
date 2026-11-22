"""Copy the production model out of the MLflow registry into serving/model/ for the Docker build.

The serving container does not reach the registry, which lives in a local SQLite file. This
script downloads the version behind `models:/inspection-ranker@production` into
serving/model/ and writes serving/model/version.json with its registry version, semantic
version, feature version, input columns, lineage tags, and the object types skops must trust to
load it. The API reports these at /health and loads the model with the trusted types.

Usage:
    python export_model.py
"""

import json
import shutil

import mlflow
import yaml

from src import config, registry, tracking


def main() -> None:
    version = registry.production_version()
    if version is None:
        raise SystemExit(f"No version of {config.MODEL_NAME} has the alias {config.PRODUCTION_ALIAS!r}. "
                         "Run pipeline.py first.")

    target = config.SERVING_MODEL_DIR
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    tracking.client()  # sets the tracking URI
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{config.MODEL_NAME}@{config.PRODUCTION_ALIAS}", dst_path=str(target))

    # The MLmodel file lists the object types that skops must trust to load this model, which the
    # registry computed when it logged the model. The serving app reads them from version.json.
    mlmodel = yaml.safe_load((target / "MLmodel").read_text())
    trusted = mlmodel["flavors"]["sklearn"].get("skops_trusted_types") or []

    tags = dict(version.tags)
    info = {
        "model_name": config.MODEL_NAME,
        "registry_version": version.version,
        "semver": tags.get("semver"),
        "feature_version": tags.get("feature_version"),
        "features": config.FEATURE_COLUMNS.get(tags.get("feature_version"), []),
        "estimator": tags.get("estimator"),
        "data_tag": tags.get("data_tag"),
        "git_commit": tags.get("git_commit"),
        "valid_headline": tags.get("valid_headline"),
        "k_week": tags.get("k_week"),
        "skops_trusted_types": trusted,
    }
    (target / "version.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Exported {config.MODEL_NAME} version {version.version} ({info['semver']}) to serving/model/")


if __name__ == "__main__":
    main()
