# Design decisions: pipeline automation and experiment tracking

This document records the decisions behind the training pipeline: how it is orchestrated, how AutoML selects a model, how a model is promoted and versioned, and how runs are tracked. Each decision lists its rationale and the alternatives considered. [data-ingestion-and-baseline.md](data-ingestion-and-baseline.md) covers the data, metric, and baselines the pipeline builds on, and the [README](../../README.md) explains how to run it.

## Outcome

The pipeline searched six algorithms on two feature versions and registered `inspection-ranker` version 1 (`1.0.0`), the v2 logistic regression baseline, under the alias `production`:

| Model | Inner holdout | Validation weekly precision@k | Standard error |
|---|---|---|---|
| Logistic regression, v2 (registered) | not searched | 0.449 | 0.023 |
| AutoML v2 (LightGBM), challenger | 0.476 | 0.447 | 0.023 |
| AutoML v1 (XGBoost) | 0.447 | 0.405 | 0.022 |

The challenger showed no detectable difference from the baseline and did not clear the gate at 0.471, so the bootstrap rule registered the baseline. [results/training/comparison.md](../../results/training/comparison.md) holds the full comparison.

## Tool responsibilities

DVC versions the data, Prefect orchestrates the steps, and MLflow tracks experiments and holds the model registry. The Prefect flow in `pipeline.py` runs `dvc repro` as its first task, then loads the splits and trains.

### Rationale

- Each tool does one job, so data versions live only in `dvc.lock` and the `data-v*` tags, and the flow never duplicates them.
- `dvc repro` skips unchanged stages, so the first task costs seconds when the data is current.
- Prefect gives each step a task with its own state and log, and the flow runs locally with a temporary server and no extra setup.

### Alternatives considered

- Airflow, which needs a scheduler, a webserver, and a metadata database for a single local flow.
- Running the data stages as Prefect tasks, which would duplicate the caching DVC already provides.

## Flow structure

`training_flow` runs six tasks: `update_data`, `load_inputs`, `search_version` (once per feature version), `choose_and_gate`, `register_model` (or `baseline_candidate` then `register_model`), and `write_reports`. Each task calls the shared `src/` modules, and the test split is never loaded.

### Rationale

- The same functions run in DVC stages, Prefect tasks, and unit tests.
- Tasks use `cache_policy=NO_CACHE`, because their inputs are large data frames and caching across runs belongs to DVC.

### Alternatives considered

- A scheduled Prefect deployment. The data is a fixed snapshot, so a schedule would rerun the same inputs, and running the flow on demand meets the automation goal.

## AutoML search

FLAML searches six algorithms (LightGBM, XGBoost, random forest, extra trees, L1 and L2 logistic regression) for 300 seconds per feature version, with seed 42. Its custom metric is 1 minus weekly precision@k, with k = 9. FLAML's built-in MLflow logging is off, and the flow logs one run per algorithm with its best configuration.

### Rationale

- The search optimizes the same measure as the baselines, the gate, and the presentation, so it selects models by failures found per week.
- Including L2 logistic regression lets the search tune the baseline's model family.
- One run per algorithm gives the MLflow comparison view one row per algorithm, which shows the selection directly.
- A fixed budget and seed bound the cost and make reruns comparable.

### Alternatives considered

- H2O AutoML, which runs a Java server and selects by its own leaderboard metrics.
- FLAML's default metric (ROC-AUC), which measures ranking across all inspections when only the top of each week is visited.

## Inner holdout for model selection

The search fits on training rows before 2023-10-30 and selects configurations by their score on the 5,910 canvass rows of the last training year (the inner holdout). The winning configuration is refit on the whole training window and scored once on the validation window. The feature version with the higher inner score supplies the challenger.

### Rationale

- Choosing the best of many configurations on one set of rows overstates that set's score. The v2 challenger scored 0.476 on the inner holdout that selected it and 0.447 on the validation window, a drop of 0.029.
- The validation window is used once, by the gate, so the gate compares the challenger and the baselines on equal terms. The baselines are also fit on the training window and scored on the validation window.
- Selection, promotion, and final evaluation each use different data: the inner holdout, the validation window, and the test year.

### Alternatives considered

- Selecting on the validation window, which inflates the score the gate reads.
- Time-series cross-validation inside FLAML. The custom metric needs each row's week, which FLAML's fold handling does not carry through.

### Limits

The search fits on 5.3 years of history in place of 6.3. The final model refits on all training and validation rows.

## Training population

Models train on every inspection type in the training window, with `inspection_group` as a feature, and are scored on canvass inspections.

### Rationale

- Complaint, license, and re-inspection visits carry history that informs later canvass visits, and dropping them halves the training data.
- Scoring on canvass rows matches the population the city can reorder.

### Alternatives considered

- Training on canvass inspections only.

## Model artifact

The logged model is an sklearn `Pipeline` of the shared preprocessing (one-hot encoding, median imputation with missing-value indicators, scaling) and the estimator. It is logged with an input signature built from the feature columns, an input example, and `pyfunc_predict_fn="predict_proba"`. MLflow stores it in the skops format, and `src/registry.py` passes the model's own non-default types as trusted.

### Rationale

- Preprocessing travels inside the model, so the API sends raw feature columns and transforms them exactly as training did.
- The signature fixes the request schema the API validates against.
- `predict_proba` returns the Fail probability that ranking needs.
- skops reloads a model only with an explicit list of trusted types. The pipeline built the model, so trusting its types (such as `numpy.dtype` from the encoder) is safe.

### Alternatives considered

- Pickle (`cloudpickle`) serialization, which loads arbitrary code.
- A separate preprocessing step in the API, which risks training-serving skew.

## Promotion gate

A challenger is registered only if its validation headline beats two bars, each raised by one standard error: the best baseline (0.449 + 0.023 = 0.471) and, when one exists, the production model's validation headline recorded when it was registered.

### Rationale

- A one-standard-error margin keeps a challenger that wins by noise out of production.
- The production model was refit on the validation rows, so rescoring it there would overstate it. Its recorded out-of-sample score is the fair comparison.

### Alternatives considered

- A gate on every slice, which multiplies the chances of a noisy failure with no requirement behind it. Slice metrics are logged for review.
- No margin over the baseline, which would promote ties.

## Bootstrap registration

When the challenger fails the gate and no production model exists, the flow registers the best logistic regression baseline as version `1.0.0`, refit on training and validation rows like any other model. The registry tag `promotion` records `challenger` or `bootstrap_baseline` for each version.

### Rationale

- The deployment stage needs a registered model, and the baseline is the best model the pipeline has validated.
- The gate stays strict: every later challenger must beat the baseline's recorded score by one standard error.

### Alternatives considered

- Relaxing the gate after seeing the result, which removes the reason for having one.
- Registering the AutoML challenger anyway, which promotes a model with no measured advantage.

## Model registry and semantic versions

MLflow's registry numbers versions 1, 2, 3. Each version carries a `semver` tag computed against the production version, and the alias `production` points at the newest version that passed the gate (or the bootstrap baseline):

| Change since the production version | Bump |
|---|---|
| Input signature (features added, removed, or retyped) | MAJOR |
| Data version (`data/processed` hash) or winning algorithm | MINOR |
| Hyperparameters or code only | PATCH |

### Rationale

- A signature change breaks the deployed API's request schema, so it is a major change.
- One alias is all the deployment stage reads, through `models:/inspection-ranker@production`.

### Alternatives considered

- MLflow's stage labels (Staging, Production, Archived), which MLflow has deprecated in favor of aliases.
- A separate `staging` alias and promotion script, which add a manual step with no reviewer in this project.

## Lineage and tracking

Every MLflow run is tagged with the git commit, whether the working tree had uncommitted changes, the nearest `data-v*` tag, the `data/processed` hash from `dvc.lock`, the facility scope, and the snapshot end. The flow logs through `MlflowClient` with explicit run IDs, under one parent run per flow run, with child runs per feature version and per algorithm. MLflow uses a local SQLite store (`mlflow.db`) and an artifact folder (`mlartifacts/`), both excluded from git.

### Rationale

- A registered model points to the exact code and data that produced it, and `git checkout` with `dvc checkout` restores both.
- Prefect may run tasks on different threads, and explicit run IDs keep each log entry on the right run.
- A local store needs no server, and anyone can open it with `mlflow ui`.

### Alternatives considered

- MLflow's fluent API with nested `start_run` contexts, which tracks the active run per thread.
- A hosted tracking server (Databricks or a shared MLflow server), which the team does not need for one shared repository.

### Limits

Each person's `mlflow.db` is local, so registered models stay on the machine that trained them. `export_model.py` copies the production model into `serving/model/` for the deployment stage.

## Handoff to deployment and monitoring

`export_model.py` writes the production model and `version.json` (registry version, semantic version, feature version, input columns, lineage tags) to `serving/model/`. The flow writes `results/reference.parquet`: the validation canvass rows with their features, labels, and the registered model's out-of-sample predictions.

### Rationale

- The serving container reads a folder, so it needs no access to the local registry.
- The reference data gives the drift monitor a baseline drawn from the same features and predictions the API will produce.

### Alternatives considered

- Building the Docker image from the registry directly with `mlflow models build-docker`, which gives less control over the API's request validation and logging.

Assisting sources: Claude Opus 5.5 (Anthropic).
