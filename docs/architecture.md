# Architecture

The system ranks Chicago restaurant inspections by predicted probability of failing, so inspectors visit the most likely failures first. It has four stages: a data pipeline versioned with DVC, a training pipeline orchestrated by Prefect and tracked in MLflow, a serving API, and drift monitoring. The data pipeline, the training pipeline, and the serving API are implemented. Monitoring is the next stage, and this document fixes the interfaces it builds on. It describes how the stages connect, where each artifact lives, and what each stage hands to the next. [design-decisions/](design-decisions/) records the reasoning behind each choice.

## Overview

```
Chicago Data Portal (4ijn-s7e5)
        |  CSV export or API, cut at 2026-10-29
        v
+---------------------------- Data pipeline (DVC) ----------------------------+
|  ingest -> validate -> clean -> features -> split          stage: prepare   |
|                                              |                              |
|                                              v                              |
|                                          baselines          stage: baselines|
+------------------------------------------------------------------------------+
        |  data/processed/{train,valid,test,drift_pre2018}.parquet
        |  results/baselines.csv, results/capacity.json
        v
+------------------------- Training pipeline (Prefect) ------------------------+
|  update_data -> load_inputs -> search_version (v1, v2) -> choose_and_gate    |
|       -> register_model -> write_reports                                     |
+------------------------------------------------------------------------------+
        |  runs, metrics, artifacts           |  model version, semver, alias
        v                                     v
   MLflow tracking                      MLflow Model Registry
                                              |  models:/inspection-ranker@production
                                              v
                                        export_model.py -> serving/model/
                                              |
                                              v
                              Serving API (Docker, FastAPI) -> prediction log
                                              |
                                              v
                    Monitoring (Evidently) against results/reference.parquet
```

## Components

| Component | Tool | Code | Responsibility |
|---|---|---|---|
| Data pipeline | DVC | `run_data.py`, `dvc.yaml`, `src/ingest.py` to `src/baselines.py` | Builds the snapshot, features, splits, and baselines; reruns only changed stages |
| Data storage | DVC remote (Google Shared Drive) | `.dvc/config`, `dvc.lock` | Holds every data version; tags `data-v1` and `data-v2` name them |
| Validation | Python rules | `src/validate.py` | Stops the pipeline on structural errors and warns on implausible values |
| Training pipeline | Prefect | `pipeline.py` | Orders the training steps and records each task's state and log |
| Model search | FLAML | `src/automl.py` | Searches six algorithms, scored by weekly precision@k |
| Experiment tracking | MLflow | `src/tracking.py` | Records parameters, metrics, artifacts, and lineage tags per run |
| Model registry | MLflow | `src/registry.py` | Applies the promotion gate, assigns semantic versions, sets the `production` alias |
| Model export | Python | `export_model.py` | Copies the production model and its metadata to `serving/model/` |
| Serving | Docker, FastAPI | `Dockerfile`, `serving/app.py` | Accepts a batch of feature rows and returns Fail probabilities, ranks, and the model's semantic version |
| Monitoring | Evidently | Monitoring stage (next) | Compares served inputs and predictions with the reference data and raises alerts |

Settings for every component live in `src/config.py`, with environment variable overrides for paths, dates, the seed, the MLflow tracking URI, and the search budget.

## Data pipeline

The `prepare` stage ingests the snapshot, validates it, keeps Pass, Pass w/ Conditions, and Fail inspections of restaurants with a linkable license (185,236 rows), builds features from strictly earlier days, and writes four fixed-date splits. The `baselines` stage computes k from inspector capacity (9 visits per week) and scores the baselines on the validation window.

| Split | Inspection dates | Role |
|---|---|---|
| `train` | 2018-07-01 to 2024-10-29 | Fitting; the last year (from 2023-10-30) is the inner holdout for model selection |
| `valid` | 2024-10-30 to 2025-10-29 | Baselines and the promotion gate |
| `test` | 2025-10-30 to 2026-10-29 | Production validation through the API; `load_split("test")` requires `allow_test=True` |
| `drift_pre2018` | before 2018-07-01 | A real drift batch under the old food code |

## Training pipeline

The Prefect flow runs `dvc repro` so the data matches the code, then searches each feature version with FLAML. The search fits on training rows before 2023-10-30 and selects on the inner holdout. The winning configuration is refit on the whole training window and scored once on the validation window. The challenger must beat the best baseline, and the current production model when one exists, by one standard error. With no production model, the best logistic regression baseline is registered. The registered model is refit on training and validation rows.

The production model is `inspection-ranker` version 1 (`1.0.0`), the v2 logistic regression, with a validation weekly precision@k of 0.449.

## Artifacts and storage

| Artifact | Location | Versioned by |
|---|---|---|
| Code, configuration, `dvc.lock`, result tables and figures | GitHub | git |
| Raw CSV, `data/raw/`, `data/processed/` | DVC cache and Google Shared Drive remote | `dvc.lock` and `data-v*` tags |
| Runs, metrics, and registered models | `mlflow.db` and `mlartifacts/` on the training machine | MLflow run IDs and registry versions |
| Exported production model | `serving/model/` on the build machine | `serving/model/version.json` |
| Monitoring reference data | `results/reference.parquet` in git | git |

The MLflow store is local to the machine that ran the flow. `export_model.py` moves the production model out of it, so the serving container needs no access to the registry.

## Interfaces between stages

The deployment and monitoring stages build on these interfaces:

| Interface | Producer | Consumer | Contents |
|---|---|---|---|
| Model package | `export_model.py` | Serving API | MLflow model with input signature and `predict_proba` output, plus `version.json` (registry version, semver, feature version, input columns, lineage tags) |
| Request schema | `config.FEATURE_COLUMNS["v2"]` | Serving API | Four categorical and six numeric feature columns, typed as in `src/train.feature_frame` |
| Input rules | `validate.validate_features()` | Serving API, monitoring | Structural errors to reject, and range warnings that flag corrupted values |
| Reference data | `register_model` task | Monitoring | Validation canvass rows with features, labels, and out-of-sample predictions |
| Evaluation data | `load_split("test", allow_test=True)` and `load_split("drift_pre2018")` | Monitoring | Clean test rows for production validation and pre-2018 rows for real drift |

Requests carry feature rows built by `src/features.py`, the same code that built the training data, so training and serving apply identical transformations.

## Reproducibility and lineage

- Data: a fixed snapshot end date, content hashes checked on every load, and DVC tags that restore any data version with `git checkout <tag>` and `dvc checkout`.
- Training: a fixed seed (42), a fixed search budget, and fixed split dates.
- Models: every MLflow run records the git commit, the `data-v*` tag, and the `data/processed` hash from `dvc.lock`, so a registered model traces back to the exact code and data that produced it.
- Environment: `requirements.txt` with minimum versions and the pins that DVC's Google Drive client needs.

Assisting sources: Claude Opus 5.5 (Anthropic).
