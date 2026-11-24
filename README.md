# Chicago Restaurant Inspection Prioritization

This project is an MLOps pipeline that helps the City of Chicago decide which restaurants to inspect first. Using the city's public food inspection records from 2010 to the present, the model predicts the probability that an establishment will fail its next inspection, based on inspection history, risk category, and recent inspections in its ZIP code. Chicago has a limited number of inspectors, so the system ranks inspections and is evaluated by how many failures it finds in the visits inspectors can make each week (weekly precision@k). The city's 2018 food code overhaul changed its violation codes, which gives the monitoring system a real schema and concept drift event to detect.

*ADSP 32021 Machine Learning Operations, University of Chicago M.S. Applied Data Science*

---

The repository implements data ingestion, baselines, the automated training pipeline with experiment tracking and a model registry, and a FastAPI service in Docker that serves the production model. Monitoring is added in the following stage. The [project proposal](docs/project_proposal.md) describes the dataset, the prediction problem, and the planned toolstack, and the [EDA notebook](eda/consolidated_eda.ipynb) records the analysis behind each data decision below. [docs/architecture.md](docs/architecture.md) shows how the stages connect and what each one hands to the next, and [docs/design-decisions/](docs/design-decisions/) records each design decision with its rationale and the alternatives considered.

## Goals

- Use a dataset with a clear target and an evaluation metric that matches the city's constraint of a fixed inspection budget.
- Split the data by time, and keep the test set isolated until the deployed model is validated in production.
- Version the data so any earlier version can be restored and compared.
- Build an automated pipeline that preprocesses the data, selects and tunes a model with AutoML, tracks every run, and registers the final model with a semantic version.
- Serve the registered model for real-time predictions, and monitor its inputs and performance for drift.

## Stack

| Role | Tool |
|---|---|
| Data processing and storage | pandas, pyarrow (parquet) |
| Ingestion from the Chicago Data Portal | requests (Socrata API) |
| Data versioning and stage caching | DVC, with a Google Drive remote |
| Baselines and preprocessing | scikit-learn |
| Orchestration | Prefect |
| AutoML | FLAML, searching LightGBM, XGBoost, random forest, extra trees, and L1 and L2 logistic regression |
| Experiment tracking and model registry | MLflow, with a local SQLite store |
| Model serialization | skops, MLflow's format for scikit-learn models |
| Serving | FastAPI, uvicorn |
| Containerization | Docker |
| Tests | pytest |
| Exploratory analysis | Jupyter, matplotlib |

## Data pipeline

`run_data.py` runs six stages in order. Each stage is a plain function in `src/`, which the training pipeline also calls. `dvc.yaml` runs the first five as the DVC stage `prepare` and the last as the stage `baselines`, so `dvc repro` reruns only the stages whose inputs changed.

```
Chicago Data Portal (4ijn-s7e5): local CSV export or API, cut at 2026-10-29
        |
        v
ingest     -> data/raw/inspections.parquet, data/raw/manifest.json (content hash)
        |
        v
validate   -> results/validation/raw.csv; a failed validation rule stops the run
        |
        v
clean      -> Pass / Pass w/ Conditions / Fail rows in the facility scope, fail target
        |
        v
features   -> history and ZIP code features from strictly earlier days
        |      results/validation/features.csv
        v
split      -> data/processed/{train,valid,test,drift_pre2018}.parquet, split_manifest.json
        |
        v
baselines  -> results/baselines.csv, baselines_slices.csv, baselines.md,
              capacity.json, metrics.json
```

## Training pipeline

`pipeline.py` is a Prefect flow that runs the data pipeline, searches for a model with AutoML, tracks every run in MLflow, and registers the winner. Each tool has one job: DVC versions the data, Prefect orders the steps and logs each run, and MLflow records experiments and holds the model registry.

```
update_data      -> dvc repro: data/processed and the baselines match the code and snapshot
        |
        v
load_inputs      -> train split (divided into inner fit and inner holdout), canvass rows of
        |           the valid split, k (capacity.json), best baseline (baselines.csv);
        |           the test split is never loaded
        v
search_version   -> per feature version (v1, v2): FLAML search on the inner split, one MLflow
        |           run per algorithm, then the winner refit on train and scored on valid
        v
choose_and_gate  -> feature version with the better inner-holdout score; its valid score must
        |           beat the best baseline and the current production model by one standard error
        v
register_model   -> refit on train + valid, log with input signature, register with a
        |           semantic version and the alias "production", save reference.parquet
        v
write_reports    -> results/training/comparison.md, comparison.csv, precision_by_k.png
```

### AutoML search

FLAML tries six algorithms, including the baselines' L2 logistic regression, within a time budget (300 seconds per feature version by default) and tunes each one's hyperparameters. Its custom metric is weekly precision@k on canvass inspections, the same measure the baselines and the gate use, so the search selects models by what the city cares about: failures found in a week's visits.

The search never sees the validation window. It fits on training rows before 2023-10-30 and scores each configuration on the canvass rows of the last training year, the inner holdout. The winning configuration is then refit on the whole training window and scored once on the validation window. Choosing the best of many configurations on one set of rows overstates that set's score, so the selection and the gate use different rows, and the validation score the gate reads is an honest estimate.

The model is an sklearn `Pipeline` of the shared preprocessing and the winning estimator, so the registered model takes the raw feature columns and transforms them the same way training did.

### Comparing feature versions

The flow searches feature version v1 (the establishment's own history) and v2 (v1 plus the ZIP code's recent fail rate and complaint count) with the same steps, budget, and outputs. The version with the higher inner-holdout headline becomes the challenger, so the validation window still plays no part in any choice. For the city, the comparison shows whether neighborhood context finds more failures per week than an establishment's own record.

### Promotion gate

A challenger is registered only if its validation headline beats the best baseline by one standard error, and also beats the production model's recorded validation headline by one standard error when a production model exists.

When the challenger fails and a production model already exists, the flow logs the reason and registers nothing. When it fails and no production model exists yet, the flow registers the best logistic regression baseline as version 1.0.0, refit on training plus validation rows like any other model. The deployment stage then has a model to serve, and every later challenger must beat that baseline's recorded score. The registry tag `promotion` records which path produced each version (`challenger` or `bootstrap_baseline`).

### Model registry and semantic versions

MLflow numbers registry versions 1, 2, 3. Each version also carries a `semver` tag that records what changed since the production version:

| Change | Version bump |
|---|---|
| Input signature (features added, removed, or retyped) | MAJOR |
| Data version or winning algorithm, same signature | MINOR |
| Hyperparameters or code only | PATCH |

The alias `production` points at the newest version that passed the gate. The deployed API loads `models:/inspection-ranker@production` through `export_model.py`.

### Lineage

Every MLflow run is tagged with the git commit, the nearest `data-v*` tag, the DVC hash of `data/processed` from `dvc.lock`, the facility scope, and the snapshot end. A registered model therefore points to the exact code and data that produced it, and `git checkout <commit>` with `dvc checkout` restores both.

## Deployment

`export_model.py` copies the production model out of the MLflow registry, and the Docker image packages it with a FastAPI app. The registry lives in a local SQLite file, so the container never connects to it and runs on any machine with Docker.

```
models:/inspection-ranker@production
        |
        v
export_model.py  -> serving/model/: model.skops, MLmodel, requirements.txt, version.json
        |
        v
docker build     -> image inspection-ranker with serving/app.py and serving/model/
        |
        v
docker run       -> FastAPI on port 8000: GET /health, POST /predict
```

### Endpoints

| Endpoint | Returns |
|---|---|
| `GET /health` | `status` and the contents of `version.json`: registry version, semantic version, feature version, input columns, estimator, data tag, git commit, validation headline, k, and the skops types trusted when loading |
| `POST /predict` | a Fail probability and a rank for each facility in the request, with the model's semantic version |
| `GET /docs` | an interactive Swagger page for both endpoints |

### Request and response

A request holds a batch of one or more facilities. Each facility has a client ID, `facility_id`, which the model does not use, and the ten v2 feature columns from the [Features](#features) table, computed with the same rules as `src/features.py`. Fields outside this list are rejected with a 422 error. The categorical values must match the values the model was trained on:

| Column | Values |
|---|---|
| `inspection_group` | `canvass`, `complaint`, `license`, `re-inspection`, `other` |
| `facility_group` | `restaurant` |
| `risk` | `high`, `medium`, `low`, `missing` |
| `prev_result` | `Pass`, `Pass w/ Conditions`, `Fail`, `first inspection` |

The preprocessing encodes a category it did not see in training as all zeros, so a misspelled value such as `High` still returns a score. `days_since_prev` and `prev_n_violations` may be null, as they are for a first inspection, and the model fills them with the training medians.

The response lists the facilities from highest to lowest Fail probability, with rank 1 for the highest. The ranking is within the batch, so a batch of one week's canvass inspections returns the order in which the city would visit them, and the top k (reported by `/health`) are that week's visits. Every response carries `model_version`, the semantic version of the model that scored it.

### Image

The Dockerfile installs `serving/requirements.txt` before copying the app, so a code change reuses the cached dependency layer. The app runs as a non-root user, and Docker checks `/health` every 30 seconds. `serving/requirements.txt` pins pandas, NumPy, scikit-learn, SciPy, and skops to the versions in the exported model's `requirements.txt`, so the container loads the model with the library versions that saved it.
## Data

The project uses the City of Chicago [Food Inspections](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5) dataset from the Chicago Data Portal (dataset ID `4ijn-s7e5`). The snapshot used here ends on 2026-10-29 and holds 317,961 inspections from 2010-01-04 onward. The CSV export is about 356 MB, which is over GitHub's file size limit, so DVC stores it and git keeps only its pointer file, `data/Food_Inspections_20261102.csv.dvc`.

The portal adds new inspections over time and occasionally revises old ones. The fixed end date removes the new rows, and the content hash in the manifest shows whether any revised rows differ from the snapshot used here.

The ingest stage writes the snapshot to `data/raw/inspections.parquet` with a manifest (`data/raw/manifest.json`) that records the source, the row count, the date range, and a SHA-256 hash of the table's contents. The split stage writes a second manifest with the facility scope and the size, fail rate, date range, and content hash of each split. Every later stage loads data through `load_snapshot()` or `load_split()`, which recompute the hash and stop if the data no longer matches its manifest. Settings live in `src/config.py`, and the random seed is fixed at 42.

## Data versioning

DVC versions the raw CSV, the raw snapshot (`data/raw/`), and the splits (`data/processed/`). `dvc repro` records the hash of every stage output in `dvc.lock`, and each data version is a git tag on the commit that holds its `dvc.lock`.

| Tag | Facility scope | Contents |
|---|---|---|
| `data-v1` | all facility types | the base version |
| `data-v2` | restaurants only | the version the model trains on |

Every version loads through the same function, `load_split()`, so code downstream never changes between versions. To restore a version, check out its tag and let DVC restore the matching data:

```bash
git checkout data-v1
dvc checkout
```

To compare the baselines of two versions, run `dvc metrics diff data-v1 data-v2`, which reads `results/metrics.json` at each tag.

## Target and modeling rows

The modeling rows are inspections with a result of Pass, Pass w/ Conditions, or Fail. The other results (Out of Business, No Entry, Not Ready, Business Not Located) record visits with no inspection outcome. Rows with a missing or zero license number are dropped, because they cannot be linked to earlier inspections.

`FACILITY_SCOPE` in `src/config.py` sets which establishments are ranked. The value `"restaurant"` keeps rows whose facility type is exactly "restaurant" after lowercasing, which matches the project's focus on restaurant inspections, and `"all"` keeps every facility type. The snapshot has 185,236 modeling rows under the `"restaurant"` scope and 272,197 under `"all"`.

The target `fail` is 1 when the result is Fail. Its yearly rate stays between 0.196 and 0.259 from 2010 to 2026, while the rate of Fail or Pass w/ Conditions more than doubles after the 2018 food code change, which would give a model several years of drift unrelated to establishment behavior.

## Features

Each feature for an inspection is computed from inspections on strictly earlier days, so no result from the same day or later reaches the model. Feature version v1 uses the establishment's own history, and v2 adds two features for its ZIP code.

| Feature | Version | Description |
|---|---|---|
| `inspection_group` | v1, v2 | canvass, complaint, license, re-inspection, or other |
| `facility_group` | v1, v2 | the 20 most common facility types, other, or missing (one value under the restaurant scope) |
| `risk` | v1, v2 | high, medium, low, or missing |
| `prev_result` | v1, v2 | result of the previous inspection, or first inspection |
| `prior_fails` | v1, v2 | Fail results on earlier days |
| `n_prior` | v1, v2 | inspections on earlier days |
| `days_since_prev` | v1, v2 | days since the previous inspection |
| `prev_n_violations` | v1, v2 | violations cited at the previous inspection |
| `zip_fail_rate_365d` | v2 | Fail share of the ZIP code's inspections in the previous 365 days, smoothed toward the training rate |
| `zip_complaints_90d` | v2 | complaint inspections in the ZIP code in the previous 90 days |

## Splits

The splits use fixed dates, so a newer export cannot move them.

| Split | Inspection dates | Use |
|---|---|---|
| `train` | 2018-07-01 to 2024-10-29 | baselines and model fitting; the AutoML search fits on rows before 2023-10-30 and selects on canvass rows from 2023-10-30 (the inner holdout) |
| `valid` | 2024-10-30 to 2025-10-29 | baseline scores and the promotion gate |
| `test` | 2025-10-30 to 2026-10-29 | production validation only |
| `drift_pre2018` | before 2018-07-01 | drift demonstration under the old food code |

Training starts at the 2018 food code change, so violation-based history uses one coding scheme. `load_split("test")` raises an error unless the caller passes `allow_test=True`, which only the production validation step does.

## Evaluation metric

The headline metric is weekly precision@k on canvass inspections. Each week, the model ranks that week's canvass inspections, and the top k are the visits the city would make first. The metric is the Fail rate among all the visits picked across the weeks of the window. Chicago has a fixed number of inspectors, so the useful question is how many failures a week's visits find, which this metric measures directly. Canvass inspections are the routine visits the city can reorder. Complaint, license, and re-inspection visits are triggered by events and take place whether or not a model ranks them.

k follows inspector capacity. It is 10% of the median number of canvass inspections per week in the training window, computed from the data and saved to `results/capacity.json` with the weekly medians, so it changes with the facility scope.

`lift_per_100` is the number of extra failures found per 100 visits at the headline, compared with random selection. Precision at 5%, 10%, and 20% of the whole window ranked at once, ROC-AUC, and PR-AUC are reported alongside. Precision@10% is also reported within slices by risk category, first or repeat inspection, North or South Side (divided at Roosevelt Road), and facility type when the scope has more than one, so a model that does well overall but poorly for one group is visible.

## Validation rules

`src/validate.py` holds two kinds of checks, which the pipeline runs on the raw snapshot and on the feature table and saves to `results/validation/`.

- Validation rules check structure: required columns, types, null and duplicate IDs, dates inside the snapshot, and allowed result values. A failure stops the pipeline.
- Verification rules check plausibility against the EDA: coordinates inside Chicago, missing-value shares, the fail rate, months with no inspections, and feature value ranges. A failure is logged as a warning.

The monitoring stage, added next, reuses the feature rules on the inputs sent to the deployed API.

## Baselines

The baselines set the bar a trained model must clear. They are scored on canvass inspections in the validation window.

- `random`: a random ranking.
- `rate:<column>`: each inspection gets the Fail rate of its category among canvass inspections in the training window, for risk, facility group, previous result, earlier fails (capped at 5), and ZIP code. A column with one value in the scope is skipped.
- `logistic_regression_v1` and `logistic_regression_v2`: a logistic regression on each feature version, trained on all training inspections.

## How to run

[docs/team-setup.md](docs/team-setup.md) condenses these steps into a short checklist for team members, with fixes for common errors.

### Prerequisites

You need Python 3.10 or later. Use a fresh virtual environment, because a conda `base` environment can hold NumPy and pandas builds that conflict and stop pandas from importing.

### Set up the environment

From the repository root, create the environment and install the dependencies, which include DVC:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

### Connect to the DVC remote

The DVC remote `gdrive` is a folder in the team's Google Shared Drive, set in `.dvc/config`. Google requires your own credentials for DVC, which stay in `.dvc/config.local` and are never committed. The team uses a Google Cloud service account that is a member of the Shared Drive with the Content manager role. Ask a teammate for its key file, save it outside the repository, and point DVC at it:

```bash
dvc remote modify --local gdrive gdrive_use_service_account true
dvc remote modify --local gdrive gdrive_service_account_json_file_path path/to/service-account.json
```

The [DVC Google Drive guide](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive) also covers signing in with your own OAuth client.

### Get the data

Choose one of the following:

- Pull it from the remote. `dvc pull` downloads the CSV, the raw snapshot, and the splits for the checked-out version.
- Download the export. Open the [dataset page](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5), choose Export, then CSV, and save the file as `data/Food_Inspections_20261102.csv`. The pipeline and the EDA notebook both read this path.
- Let the pipeline download it. When the CSV is missing, `python run_data.py` downloads the same rows from the portal API, filtered to inspection dates up to 2026-10-29. `dvc repro` needs the CSV, so use one of the first two options with it.

### Run the tests

```bash
python -m pytest
```

All 42 tests should pass.

### Run the data pipeline

```bash
dvc repro
```

From the local CSV, the run takes about a minute. `dvc repro` skips stages whose inputs have not changed. Without DVC, `python run_data.py` runs every stage; add `--source api` to download from the portal, `--skip-baselines` to stop after the splits, or `--baselines-only` to rescore existing splits.

### Check the run

A run on the same snapshot reproduces the numbers in the Results section. The log should report the modeling rows and the split sizes below, with no validation warnings, and `results/baselines.md` should match the committed copy. For an exact comparison, `dvc status` reports no changes when every output matches `dvc.lock`.

### Run the training pipeline

```bash
python pipeline.py
```

The flow runs `dvc repro`, searches both feature versions for 300 seconds each, and registers the winner if it passes the gate (or the best baseline on the first run, as described under Promotion gate), so a full run takes about 11 minutes. Prefect starts a temporary local server for the run, so no separate Prefect setup is needed. Options:

- `--time-budget 60` shortens the search, for a quick check
- `--feature-versions v2` searches one feature version
- `--no-register` searches and compares without registering
- `--skip-dvc` uses the data already in `data/processed`

### Inspect runs and the registry

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`python -m` runs the MLflow installed in `.venv`. A bare `mlflow` command can resolve to another installation, such as one in conda `base`, which cannot read the database this project's MLflow version creates. Open http://127.0.0.1:5000. The experiment `inspection-ranking` holds one parent run per flow run, a child run per feature version, and a run per algorithm with its best configuration and score. Select the algorithm runs and choose Compare to see the AutoML selection side by side. The Models page shows `inspection-ranker` with its versions, `semver` tags, and the `production` alias.

### Export the production model

```bash
python export_model.py
```

This copies the production model and a `version.json` summary into `serving/model/` for the deployment stage.

### Build and run the API

Docker Desktop must be running. Build the image from the repository root after exporting the model. The image copies `serving/model/` at build time, so rebuild it after every export.

```bash
docker build -t inspection-ranker .
```

```bash
docker run --rm -p 8000:8000 inspection-ranker
```

The log ends with `Uvicorn running on http://0.0.0.0:8000`. Open http://localhost:8000/docs to try the endpoints, or send requests from a second terminal:

```bash
curl http://localhost:8000/health
```

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"facilities": [{"facility_id": "A", "inspection_group": "canvass", "facility_group": "restaurant", "risk": "high", "prev_result": "Fail", "prior_fails": 3, "n_prior": 8, "days_since_prev": 120, "prev_n_violations": 6, "zip_fail_rate_365d": 0.28, "zip_complaints_90d": 15}, {"facility_id": "B", "inspection_group": "canvass", "facility_group": "restaurant", "risk": "low", "prev_result": "first inspection", "prior_fails": 0, "n_prior": 0, "days_since_prev": null, "prev_n_violations": null, "zip_fail_rate_365d": 0.18, "zip_complaints_90d": 2}]}'
```

With version 1.0.0, `/health` reports `"semver": "1.0.0"` and `"k_week": "9"`, and `/predict` ranks facility A first with a score of 0.429 and facility B second with 0.249. Press Ctrl+C to stop the container, which `--rm` then removes.

## Results

The results below are for data version `data-v2` (restaurants only), the version the model trains on.

### Splits

| Split | Inspections | Canvass inspections | Fail rate | Canvass fail rate |
|---|---|---|---|---|
| `train` | 62,070 | 30,444 | 0.229 | 0.247 |
| `valid` | 11,541 | 6,099 | 0.211 | 0.228 |
| `test` | 9,749 | 4,504 | 0.208 | 0.222 |
| `drift_pre2018` | 101,876 | 50,401 | 0.211 | 0.208 |

### Inspector capacity

In the training window, restaurants received a median of 191 inspections per week, 93 of them canvass inspections. k is 10% of the canvass figure, 9 visits per week, which picks 477 inspections over the 53 weeks of the validation window.

### Baselines

Scored on 6,099 canvass inspections in the validation window, with a base rate of 0.228. The standard error of weekly precision@k is between 0.018 and 0.023 for every baseline.

| Baseline | Weekly precision@k | Lift per 100 | ROC-AUC |
|---|---|---|---|
| `logistic_regression_v2` | 0.449 | 22.1 | 0.680 |
| `logistic_regression_v1` | 0.394 | 16.6 | 0.646 |
| `rate:prior_fails_capped` | 0.335 | 10.8 | 0.539 |
| `rate:zip` | 0.287 | 5.9 | 0.546 |
| `rate:prev_result` | 0.281 | 5.3 | 0.545 |
| `rate:risk` | 0.237 | 0.9 | 0.508 |
| `random` | 0.197 | -3.1 | 0.486 |

The v2 logistic regression is the strongest baseline, 0.055 above v1, so the ZIP code features add lift. Risk shows no detectable lift, and random ranking sits within two standard errors of the base rate. Facility type has one value under this scope, so its rate baseline is skipped. The full table, with precision at 5%, 10%, and 20% of the window and PR-AUC, is in [results/baselines.md](results/baselines.md), and precision@10% by slice is in [results/baselines_slices.csv](results/baselines_slices.csv).

### Data versions compared

The two versions differ only in facility scope: `data-v1` keeps every facility type, and `data-v2` keeps restaurants. Both use the same snapshot, split dates, features, and code.

| | `data-v1` (all facility types) | `data-v2` (restaurants) |
|---|---|---|
| Modeling rows | 272,197 | 185,236 |
| Canvass inspections in validation | 8,556 | 6,099 |
| Median canvass inspections per week (train) | 136 | 93 |
| k (visits per week) | 14 | 9 |
| Canvass base rate (validation) | 0.223 | 0.228 |
| `logistic_regression_v2`, weekly precision@k | 0.430 | 0.449 |
| `logistic_regression_v1`, weekly precision@k | 0.394 | 0.394 |
| `random`, weekly precision@k | 0.228 | 0.197 |

The best baseline scores 0.019 higher on restaurants, which is within one standard error, so the scope change shows no detectable difference in baseline quality. `dvc metrics diff data-v1 data-v2` prints the same comparison for every baseline.

### Algorithm comparison

Each algorithm's best configuration, scored on 5,910 canvass inspections in the inner holdout (the last training year), with a 300-second search per feature version:

| Algorithm | v1 | v2 |
|---|---|---|
| LightGBM (`lgbm`) | 0.444 | 0.476 |
| XGBoost (`xgboost`) | 0.447 | 0.465 |
| Random forest (`rf`) | 0.415 | 0.455 |
| L1 logistic regression (`lrl1`) | 0.411 | 0.444 |
| L2 logistic regression (`lrl2`) | 0.405 | 0.444 |
| Extra trees (`extra_tree`) | 0.411 | 0.442 |

Every algorithm scores higher with v2, so the ZIP code features help across model families. v2's inner score (0.476) beats v1's (0.447), so the v2 LightGBM model became the challenger. MLflow holds one run per algorithm with its configuration, which the MLflow UI compares side by side.

### Registered model

Each feature version's winner, refit on the whole training window and scored once on 6,099 canvass inspections in the validation window:

| Model | Inner holdout | Validation weekly precision@k | Standard error | Lift per 100 | ROC-AUC |
|---|---|---|---|---|---|
| Best baseline (`logistic_regression_v2`) | not searched | 0.449 | 0.023 | 22.1 | 0.680 |
| AutoML v2 (LightGBM) | 0.476 | 0.447 | 0.023 | 21.9 | 0.672 |
| AutoML v1 (XGBoost) | 0.447 | 0.405 | 0.022 | 17.7 | 0.646 |

The challenger scored 0.029 lower on the validation window than on the inner holdout that selected it, which is the optimism the separate holdout removes. On the validation window it shows no detectable difference from the logistic regression baseline (0.447 against 0.449), so it does not clear the gate at 0.471. With no production model yet, the flow registered the v2 logistic regression baseline, refit on training plus validation rows, as `inspection-ranker` version 1 (`1.0.0`) with the `production` alias. A later challenger has to beat 0.449 by one standard error to replace it.

For the city, the result means a simple logistic regression on inspection history and ZIP code context finds about 22 more failures per 100 routine visits than random selection, and a broad AutoML search found nothing measurably better on this data. [results/training/comparison.md](results/training/comparison.md) holds the comparison, and [results/training/precision_by_k.png](results/training/precision_by_k.png) plots weekly precision across values of k.

## Outputs

| Path | Contents |
|---|---|
| `data/raw/inspections.parquet` | raw snapshot in snake_case columns (DVC) |
| `data/raw/manifest.json` | source, row count, date range, and content hash of the snapshot (DVC) |
| `data/processed/<split>.parquet` | feature table for each split (DVC) |
| `data/processed/split_manifest.json` | facility scope, split dates, feature lists, and size, fail rate, and hash of each split (DVC) |
| `dvc.lock` | hashes of every stage's dependencies and outputs for the checked-out version |
| `results/validation/raw.csv` | validation and verification report for the raw snapshot |
| `results/validation/features.csv` | validation and verification report for the feature table |
| `results/baselines.csv` | weekly precision@k, precision@k shares, lift, ROC-AUC, and PR-AUC for each baseline |
| `results/baselines_slices.csv` | precision@10% for each baseline within each slice |
| `results/baselines.md` | summary table of the baselines |
| `results/capacity.json` | median inspections per week in the training window and the k they imply |
| `results/metrics.json` | headline metric per baseline, read by `dvc metrics diff` |
| `results/training/automl_summary_<version>.csv` | best inner-holdout headline and configuration of each algorithm, per feature version |
| `results/training/slices_<version>.csv` | precision@10% within each slice for the best model of each feature version |
| `results/training/feature_importance_<version>.png` | feature importances or coefficients of the best model of each feature version |
| `results/training/comparison.md`, `comparison.csv` | AutoML winners against the best baseline, with the gate decision |
| `results/training/precision_by_k.png` | weekly precision@k across k for each feature version's best model |
| `results/reference.parquet` | validation canvass rows with features, labels, and the registered model's out-of-sample predictions, the monitoring reference |
| `mlflow.db`, `mlartifacts/` | MLflow runs, registry, and logged models (not committed) |
| `serving/model/` | exported production model for the deployment stage (not committed) |

## Repository layout

```
.
├── README.md
├── requirements.txt
├── pytest.ini
├── dvc.yaml                 # DVC stages: prepare and baselines
├── dvc.lock                 # output hashes for the checked-out data version
├── run_data.py              # entry point for the data ingestion and baseline stage
├── pipeline.py              # Prefect training flow: AutoML, MLflow tracking, registry
├── export_model.py          # copies the production model into serving/model/
├── Dockerfile               # serving image: FastAPI app and the exported model
├── .dockerignore
├── serving/
│   ├── app.py               # FastAPI app: /health and /predict
│   ├── requirements.txt     # serving dependencies, pinned to the model's versions
│   └── model/               # exported production model (written by export_model.py, not committed)
├── src/
│   ├── config.py            # paths, facility scope, split dates, metric settings, feature lists
│   ├── manifest.py          # content hashes and manifest files
│   ├── ingest.py            # local CSV or portal API, raw snapshot and manifest
│   ├── validate.py          # validation (hard-fail) and verification (warning) rules
│   ├── clean.py             # modeling rows, facility scope, target, cleaned categories
│   ├── features.py          # history and ZIP code features, shared preprocessing
│   ├── split.py             # fixed-date splits, split manifest, guarded loader
│   ├── metrics.py           # weekly precision@k, capacity k, lift, slice metrics
│   ├── baselines.py         # baseline scoring and result tables
│   ├── tracking.py          # MLflow setup and lineage tags
│   ├── train.py             # model assembly, evaluation, refit, figures
│   ├── automl.py            # FLAML search with the weekly precision@k metric
│   └── registry.py          # promotion gate, semantic versions, registration
├── tests/                   # unit tests for features, splits, validation, metrics, AutoML, registry
├── docs/
│   ├── architecture.md      # stages, artifacts, and the interfaces between them
│   ├── project_proposal.md
│   ├── team-setup.md        # setup checklist for team members
│   └── design-decisions/    # decisions, rationale, and alternatives for each stage
├── eda/
│   └── consolidated_eda.ipynb
├── final_project_mpiira/    # notebook version of the ingestion stage, with restaurant EDA
├── data/                    # DVC pointer files; the data itself is in DVC
└── results/
    ├── baselines.csv
    ├── baselines.md
    ├── baselines_slices.csv
    ├── capacity.json
    ├── metrics.json
    ├── reference.parquet
    ├── training/
    └── validation/
```

## Assisting sources

Claude Opus 5.5 (Anthropic).
