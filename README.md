# Chicago Restaurant Inspection Prioritization

This project is an MLOps pipeline that helps the City of Chicago decide which restaurants to inspect first. Using the city's public food inspection records from 2010 to the present, the model predicts the probability that an establishment will fail its next inspection, based on inspection history, risk category, and recent inspections in its ZIP code. Chicago has a limited number of inspectors, so the system ranks inspections and is evaluated by how many failures it finds in the visits inspectors can make each week (weekly precision@k). The city's 2018 food code overhaul changed its violation codes, which gives the monitoring system a real schema and concept drift event to detect.

*ADSP 32021 Machine Learning Operations, University of Chicago M.S. Applied Data Science*

---

The repository implements the data ingestion and baseline stage. Pipeline orchestration, experiment tracking, deployment, and monitoring are added in the following stages. The [project proposal](docs/project_proposal.md) describes the dataset, the prediction problem, and the planned toolstack, and the [EDA notebook](eda/eda_rios.ipynb) records the analysis behind each data decision below.

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
| Tests | pytest |
| Exploratory analysis | Jupyter, matplotlib |

## Pipeline

`run_data.py` runs six stages in order. Each stage is a plain function in `src/`, so the orchestrated pipeline in the next stage can call the same code. `dvc.yaml` runs the first five as the DVC stage `prepare` and the last as the stage `baselines`, so `dvc repro` reruns only the stages whose inputs changed.

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

`FACILITY_SCOPE` in `src/config.py` sets which establishments are ranked. The value `"restaurant"` keeps rows whose facility type is exactly "restaurant" after lowercasing, which matches the project's focus on restaurant inspections, and `"all"` keeps every facility type. Under the `"all"` scope, the snapshot has 272,197 modeling rows.

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
| `train` | 2018-07-01 to 2024-10-29 | model fitting |
| `valid` | 2024-10-30 to 2025-10-29 | baselines and model selection |
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

The deployed API and the monitoring stage reuse the feature rules.

## Baselines

The baselines set the bar a trained model must clear. They are scored on canvass inspections in the validation window.

- `random`: a random ranking.
- `rate:<column>`: each inspection gets the Fail rate of its category among canvass inspections in the training window, for risk, facility group, previous result, earlier fails (capped at 5), and ZIP code. A column with one value in the scope is skipped.
- `logistic_regression_v1` and `logistic_regression_v2`: a logistic regression on each feature version, trained on all training inspections.

## How to run

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

The DVC remote `gdrive` is a shared Google Drive folder, set in `.dvc/config`. Google requires your own credentials for DVC, which stay in `.dvc/config.local` and are never committed. With a Google Cloud service account that the folder is shared with, point DVC at its key file:

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

All 29 tests should pass.

### Run the pipeline

```bash
dvc repro
```

From the local CSV, the run takes about a minute. `dvc repro` skips stages whose inputs have not changed. Without DVC, `python run_data.py` runs every stage; add `--source api` to download from the portal, `--skip-baselines` to stop after the splits, or `--baselines-only` to rescore existing splits.

### Check the run

A run on the same snapshot reproduces the numbers in the Results section. The log should report the modeling rows and the split sizes below, with no validation warnings, and `results/baselines.md` should match the committed copy. For an exact comparison, `dvc status` reports no changes when every output matches `dvc.lock`.

## Results

### Splits

| Split | Inspections | Canvass inspections | Fail rate | Canvass fail rate |
|---|---|---|---|---|
| `train` | 91,344 | 44,596 | 0.233 | 0.247 |
| `valid` | 16,208 | 8,556 | 0.210 | 0.223 |
| `test` | 13,856 | 6,435 | 0.208 | 0.217 |
| `drift_pre2018` | 150,789 | 72,805 | 0.222 | 0.214 |

### Baselines

Scored on 8,556 canvass inspections in the validation window, with a base rate of 0.223. The standard error of precision@10% is between 0.014 and 0.017 for every baseline.

| Baseline | Precision@10% | Lift per 100 | ROC-AUC |
|---|---|---|---|
| `logistic_regression_v2` | 0.425 | 20.2 | 0.672 |
| `logistic_regression_v1` | 0.378 | 15.5 | 0.638 |
| `rate:zip` | 0.318 | 9.5 | 0.542 |
| `rate:prior_fails_capped` | 0.292 | 7.0 | 0.546 |
| `rate:prev_result` | 0.281 | 5.8 | 0.542 |
| `rate:risk` | 0.244 | 2.2 | 0.504 |
| `rate:facility_group` | 0.233 | 1.0 | 0.515 |
| `random` | 0.232 | 0.9 | 0.498 |

The v2 logistic regression is the strongest baseline. Its precision@10% is 0.047 above v1, about 2.8 standard errors, so the ZIP code features add lift. Risk and facility type show no detectable lift over random ranking, and their ROC-AUC is close to 0.5. The full table, with precision at 5% and 20% and PR-AUC, is in [results/baselines.md](results/baselines.md), and precision@10% by slice is in [results/baselines_slices.csv](results/baselines_slices.csv).

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

## Repository layout

```
.
├── README.md
├── requirements.txt
├── pytest.ini
├── dvc.yaml                 # DVC stages: prepare and baselines
├── dvc.lock                 # output hashes for the checked-out data version
├── run_data.py              # entry point for the data ingestion and baseline stage
├── src/
│   ├── config.py            # paths, facility scope, split dates, metric settings, feature lists
│   ├── manifest.py          # content hashes and manifest files
│   ├── ingest.py            # local CSV or portal API, raw snapshot and manifest
│   ├── validate.py          # validation (hard-fail) and verification (warning) rules
│   ├── clean.py             # modeling rows, facility scope, target, cleaned categories
│   ├── features.py          # history and ZIP code features, shared preprocessing
│   ├── split.py             # fixed-date splits, split manifest, guarded loader
│   ├── metrics.py           # weekly precision@k, capacity k, lift, slice metrics
│   └── baselines.py         # baseline scoring and result tables
├── tests/                   # unit tests for features, splits, validation, and metrics
├── docs/
│   └── project_proposal.md
├── eda/
│   └── eda_rios.ipynb
├── final_project_mpiira/    # notebook version of the ingestion stage, with restaurant EDA
├── data/                    # DVC pointer files; the data itself is in DVC
└── results/
    ├── baselines.csv
    ├── baselines.md
    ├── baselines_slices.csv
    ├── capacity.json
    ├── metrics.json
    └── validation/
```

## Assisting sources

Claude Opus 5.5 (Anthropic).
