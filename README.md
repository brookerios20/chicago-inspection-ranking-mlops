# Chicago Restaurant Inspection Prioritization

This project is an MLOps pipeline that helps the City of Chicago decide which restaurants to inspect first. Using the city's public food inspection records from 2010 to the present, the model predicts the probability that an establishment will fail its next inspection, based on inspection history, facility type, risk category, and recent inspections in its ZIP code. Chicago has a limited number of inspectors, so the system ranks establishments and is evaluated by how many failures it finds in the top k visits (precision@k). The city's 2018 food code overhaul changed its violation codes, which gives the monitoring system a real schema and concept drift event to detect.

*ADSP 32021 Machine Learning Operations, University of Chicago M.S. Applied Data Science*

---

The repository implements the data ingestion and baseline stage. Pipeline orchestration, experiment tracking, deployment, and monitoring are added in the following stages. The [project proposal](docs/project_proposal.md) describes the dataset, the prediction problem, and the planned toolstack, and the [EDA notebook](eda/eda_rios.ipynb) records the analysis behind each data decision below.

## Goals

- Use a dataset with a clear target and an evaluation metric that matches the city's constraint of a fixed inspection budget.
- Split the data by time, and keep the test set isolated until the deployed model is validated in production.
- Build an automated pipeline that preprocesses the data, selects and tunes a model with AutoML, tracks every run, and registers the final model with a semantic version.
- Serve the registered model for real-time predictions, and monitor its inputs and performance for drift.

## Stack

| Role | Tool |
|---|---|
| Data processing and storage | pandas, pyarrow (parquet) |
| Ingestion from the Chicago Data Portal | requests (Socrata API) |
| Baselines and preprocessing | scikit-learn |
| Tests | pytest |
| Exploratory analysis | Jupyter, matplotlib |

## Pipeline

`run_data.py` runs six stages in order. Each stage is a plain function in `src/`, so the orchestrated pipeline in the next stage can call the same code.

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
clean      -> Pass / Pass w/ Conditions / Fail rows, fail target, cleaned categories
        |
        v
features   -> history and ZIP code features from strictly earlier days
        |      results/validation/features.csv
        v
split      -> data/processed/{train,valid,test,drift_pre2018}.parquet, split_manifest.json
        |
        v
baselines  -> results/baselines.csv, baselines_slices.csv, baselines.md
```

## Data

The project uses the City of Chicago [Food Inspections](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5) dataset from the Chicago Data Portal (dataset ID `4ijn-s7e5`). The snapshot used here ends on 2026-10-29 and holds 317,961 inspections from 2010-01-04 onward. The CSV export is about 356 MB, which is over GitHub's file size limit, so it is kept out of the repository. The How to run section explains how to get it.

The portal adds new inspections over time and occasionally revises old ones. The fixed end date removes the new rows, and the content hash in the manifest shows whether any revised rows differ from the snapshot used here.

The ingest stage writes the snapshot to `data/raw/inspections.parquet` with a manifest (`data/raw/manifest.json`) that records the source, the row count, the date range, and a SHA-256 hash of the table's contents. The split stage writes a second manifest with the size, fail rate, date range, and content hash of each split. Every later stage loads data through `load_snapshot()` or `load_split()`, which recompute the hash and stop if the data no longer matches its manifest. Settings live in `src/config.py`, and the random seed is fixed at 42.

## Target and modeling rows

The modeling rows are inspections with a result of Pass, Pass w/ Conditions, or Fail. The other results (Out of Business, No Entry, Not Ready, Business Not Located) record visits with no inspection outcome. Rows with a missing or zero license number are dropped, because they cannot be linked to earlier inspections, which leaves 272,197 modeling rows.

The target `fail` is 1 when the result is Fail. Its yearly rate stays between 0.196 and 0.259 from 2010 to 2026, while the rate of Fail or Pass w/ Conditions more than doubles after the 2018 food code change, which would give a model several years of drift unrelated to establishment behavior.

## Features

Each feature for an inspection is computed from inspections on strictly earlier days, so no result from the same day or later reaches the model. Feature version v1 uses the establishment's own history, and v2 adds two features for its ZIP code.

| Feature | Version | Description |
|---|---|---|
| `inspection_group` | v1, v2 | canvass, complaint, license, re-inspection, or other |
| `facility_group` | v1, v2 | the 20 most common facility types, other, or missing |
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

The headline metric is precision@10% on canvass inspections: the Fail rate among the 10% of canvass inspections the model ranks highest. Chicago has a fixed number of inspectors, so the useful question is how many failures a given number of visits finds, which precision@k measures directly. Canvass inspections are the routine visits the city can reorder. Complaint, license, and re-inspection visits are triggered by events and take place whether or not a model ranks them.

Precision at 5% and 20%, ROC-AUC, and PR-AUC are reported alongside the headline. `lift_per_100` is the number of extra failures found per 100 inspections at the headline k, compared with random selection. Precision@10% is also reported within slices by risk category, first or repeat inspection, North or South Side (divided at Roosevelt Road), and facility type, so a model that does well overall but poorly for one group is visible.

## Validation rules

`src/validate.py` holds two kinds of checks, which the pipeline runs on the raw snapshot and on the feature table and saves to `results/validation/`.

- Validation rules check structure: required columns, types, null and duplicate IDs, dates inside the snapshot, and allowed result values. A failure stops the pipeline.
- Verification rules check plausibility against the EDA: coordinates inside Chicago, missing-value shares, the fail rate, months with no inspections, and feature value ranges. A failure is logged as a warning.

The deployed API and the monitoring stage reuse the feature rules.

## Baselines

The baselines set the bar a trained model must clear. They are scored on canvass inspections in the validation window.

- `random`: a random ranking.
- `rate:<column>`: each inspection gets the Fail rate of its category among canvass inspections in the training window, for risk, facility group, previous result, earlier fails (capped at 5), and ZIP code.
- `logistic_regression_v1` and `logistic_regression_v2`: a logistic regression on each feature version, trained on all training inspections.

## How to run

### Prerequisites

You need Python 3.10 or later. Use a fresh virtual environment, because a conda `base` environment can hold NumPy and pandas builds that conflict and stop pandas from importing.

### Set up the environment

From the repository root, create the environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

### Get the data

Choose one of the following:

- Download the export. Open the [dataset page](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5), choose Export, then CSV, and save the file as `data/Food_Inspections_20261102.csv`. The pipeline and the EDA notebook both read this path.
- Let the pipeline download it. When the CSV is missing, `run_data.py` downloads the same rows from the portal API, filtered to inspection dates up to 2026-10-29.

### Run the tests

```bash
python -m pytest
```

All 25 tests should pass.

### Run the pipeline

```bash
python run_data.py
```

From the local CSV, the run takes about a minute. Add `--source api` to download from the portal even when the CSV exists, or `--skip-baselines` to stop after the splits are written.

### Check the run

A run on the same snapshot reproduces the numbers in the Results section. The log should report 272,197 modeling rows and the split sizes below, with no validation warnings, and `results/baselines.md` should match the committed copy. For an exact comparison, the `content_sha256` values in `data/processed/split_manifest.json` should match across machines.

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
| `data/raw/inspections.parquet` | raw snapshot in snake_case columns (not committed) |
| `data/raw/manifest.json` | source, row count, date range, and content hash of the snapshot (not committed) |
| `data/processed/<split>.parquet` | feature table for each split (not committed) |
| `data/processed/split_manifest.json` | split dates, feature lists, and size, fail rate, and hash of each split (not committed) |
| `results/validation/raw.csv` | validation and verification report for the raw snapshot |
| `results/validation/features.csv` | validation and verification report for the feature table |
| `results/baselines.csv` | precision@k, lift, ROC-AUC, and PR-AUC for each baseline |
| `results/baselines_slices.csv` | precision@10% for each baseline within each slice |
| `results/baselines.md` | summary table of the baselines |

## Repository layout

```
.
├── README.md
├── requirements.txt
├── pytest.ini
├── run_data.py              # entry point for the data ingestion and baseline stage
├── src/
│   ├── config.py            # paths, snapshot and split dates, metric settings, feature lists
│   ├── manifest.py          # content hashes and manifest files
│   ├── ingest.py            # local CSV or portal API, raw snapshot and manifest
│   ├── validate.py          # validation (hard-fail) and verification (warning) rules
│   ├── clean.py             # modeling rows, target, cleaned categories
│   ├── features.py          # history and ZIP code features, shared preprocessing
│   ├── split.py             # fixed-date splits, split manifest, guarded loader
│   ├── metrics.py           # precision@k, lift, slice metrics
│   └── baselines.py         # baseline scoring and result tables
├── tests/                   # unit tests for features, splits, validation, and metrics
├── docs/
│   └── project_proposal.md
├── eda/
│   └── eda_rios.ipynb
├── data/                    # not committed
└── results/
    ├── baselines.csv
    ├── baselines.md
    ├── baselines_slices.csv
    └── validation/
```

## Assisting sources

Claude Opus 5.5 (Anthropic).
