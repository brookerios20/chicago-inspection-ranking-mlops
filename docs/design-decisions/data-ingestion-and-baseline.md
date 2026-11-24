# Design decisions: data ingestion and baselines

This document records the decisions behind the data stage: how the snapshot is loaded and versioned, which rows and target the model uses, how features and splits are built, how models are evaluated, and which baselines a trained model must beat. Each decision lists its rationale and the alternatives considered. The [README](../../README.md) explains how to run the stage, and [pipeline-automation-and-experiment-tracking.md](pipeline-automation-and-experiment-tracking.md) covers training and the model registry.

## Data snapshot

The pipeline reads the City of Chicago Food Inspections dataset (`4ijn-s7e5`) from a local CSV export or from the portal API, and keeps inspections dated up to 2026-10-29. The snapshot holds 317,961 inspections from 2010-01-04 onward. Ingest writes `data/raw/inspections.parquet` and a manifest with the source, row count, date range, and a SHA-256 hash of the table's contents.

### Rationale

- A fixed end date makes a later export return the same rows, so results reproduce.
- The content hash detects rows the portal revised after the snapshot, and every later stage checks it before loading data.
- The API path lets anyone rebuild the snapshot without a manual download.

### Alternatives considered

- Reading whatever the latest export contains, which moves every count each time the portal updates.

## Data versioning

DVC versions the CSV, `data/raw/`, and `data/processed/`. `dvc.yaml` defines two stages, `prepare` (ingest to splits) and `baselines`, and `dvc.lock` records the hash of every output. Each data version is a git tag on the commit holding its `dvc.lock`: `data-v1` (all facility types) and `data-v2` (restaurants only). The remote is a folder in the team's Google Shared Drive, reached through a Google Cloud service account.

### Rationale

- Any version can be restored with `git checkout <tag>` and `dvc checkout`, and every version loads through the same function, `load_split()`.
- `dvc metrics diff data-v1 data-v2` compares the baselines of two versions directly.
- `dvc repro` reruns only the stages whose inputs changed.
- A Shared Drive accepts the service account as a member, so one key gives the team access.

### Alternatives considered

- Committing the data to git. The CSV is 356 MB, over GitHub's 100 MB file limit.
- Content-hash manifests alone, which detect changes but cannot restore an earlier version.
- A local DVC cache with no remote, where teammates rebuild the data with `dvc repro`. This remains the fallback, documented in [team-setup.md](../team-setup.md).

### Constraints

DVC's Google Drive client (PyDrive2) works only with `pyOpenSSL` 24.2.1 and `cryptography` below 44, so `requirements.txt` pins both.

## Target

The target `fail` is 1 when an inspection's result is Fail.

### Rationale

- The yearly Fail rate stays between 0.196 and 0.259 from 2010 to 2026.
- The rate of Fail or Pass w/ Conditions more than doubles after the July 2018 food code change and takes until about 2025 to return to its earlier level, which would give a model years of drift unrelated to establishment behavior ([EDA notebook](../../eda/consolidated_eda.ipynb)).

### Alternatives considered

- Fail or Pass w/ Conditions as a combined target.

## Modeling rows and facility scope

The modeling rows are inspections with a result of Pass, Pass w/ Conditions, or Fail, a linkable license number, and the facility type "restaurant". This leaves 185,236 rows. `FACILITY_SCOPE` in `src/config.py` selects the scope, and `data-v1` keeps every facility type (272,197 rows) for comparison.

### Rationale

- Out of Business, No Entry, Not Ready, and Business Not Located record visits with no inspection outcome.
- A missing or zero license cannot be linked to earlier inspections, so history features cannot be built.
- The restaurant scope matches the problem statement. On the validation window, the best baseline scores 0.449 on restaurants and 0.430 on all facility types, a difference within one standard error, so the scope narrows the problem without a measurable cost.

### Alternatives considered

- All facility types, kept as `data-v1`.

## Features

Every feature for an inspection is computed from inspections on strictly earlier days. Feature version v1 holds the establishment's history and categories: inspection group, facility group, risk, previous result, earlier fails, earlier inspections, days since the previous inspection, and violations at the previous inspection. Version v2 adds the ZIP code's Fail rate over the previous 365 days, smoothed toward the training rate, and its complaint inspections over the previous 90 days.

### Rationale

- A re-inspection can follow a Fail on the same day, so a same-day result would leak the outcome of the visit being predicted.
- Results and dates are unaffected by the 2018 renumbering of violation codes, and the violation count is the only code-based feature.
- The two versions test whether neighborhood context adds lift. It does: the v2 logistic regression baseline scores 0.449 against 0.394 for v1.

### Alternatives considered

- A feature store with point-in-time joins. The model receives rows built by the same `src/features.py` in training and serving, which avoids training-serving skew without a separate service.
- Raw ZIP code as a category, which the trailing ZIP Fail rate replaces with a value that updates over time.

## Splits and test isolation

The splits use fixed dates:

| Split | Inspection dates | Rows | Canvass rows |
|---|---|---|---|
| `train` | 2018-07-01 to 2024-10-29 | 62,070 | 30,444 |
| `valid` | 2024-10-30 to 2025-10-29 | 11,541 | 6,099 |
| `test` | 2025-10-30 to 2026-10-29 | 9,749 | 4,504 |
| `drift_pre2018` | before 2018-07-01 | 101,876 | 50,401 |

`load_split("test")` raises an error unless the caller passes `allow_test=True`, which only production validation does.

### Rationale

- Time-based splits match how the model is used: trained on the past, scored on later inspections.
- Training starts at the 2018 food code change, so violation-based history uses one coding scheme.
- The pre-2018 batch gives the monitoring stage a real shift in the violation coding to detect.
- A guard in code keeps the test year out of training and selection without relying on convention.

### Alternatives considered

- Random splits, which mix future inspections into training.
- Split dates computed from the latest date in the export, which move with each new export.

## Validation rules

`src/validate.py` holds two rule sets, run on the raw snapshot and on the feature table. Validation rules check structure (required columns, types, null and duplicate IDs, dates inside the snapshot, allowed result values) and stop the pipeline. Verification rules check plausibility against the EDA (coordinates inside Chicago, missing-value shares, the fail rate, months without inspections, feature value ranges) and log warnings.

### Rationale

- A structural failure makes every later output wrong, so the pipeline stops. A plausibility failure may be a real change in the data, so it is reported for review.
- The deployed API and the monitoring stage reuse the feature rules to catch corrupted inputs.

### Alternatives considered

- Great Expectations or TensorFlow Data Validation. The rules are few and specific to this dataset, so plain functions with one report format keep them readable and easy to reuse in the API.

## Evaluation metric

The headline metric is weekly precision@k on canvass inspections. Each week, the model ranks that week's canvass inspections, the top k are scored, and the metric is the Fail rate among all picks across the window. k is 10% of the median number of canvass inspections per week in the training window. For restaurants, the median is 93 canvass inspections (191 of all types), so k is 9, and the validation window has 477 picks. Precision at 5%, 10%, and 20% of the window, ROC-AUC, PR-AUC, and lift per 100 visits are reported alongside, and precision@10% is reported within slices.

### Rationale

- The city has a fixed number of inspectors, so the useful measure is how many failures a week's visits find.
- Canvass inspections are the routine visits the city can reorder. Complaint, license, and re-inspection visits are triggered by events.
- Computing k from the data ties it to inspector capacity and updates it when the scope or snapshot changes.
- Slices by risk, inspection history, region, and facility type show a model that does well overall but poorly for one group.

### Alternatives considered

- Precision@10% over the whole window ranked at once, kept as a secondary metric.
- ROC-AUC as the headline, which rewards ranking across all inspections when only the top of each week is visited.

### Limits

With 477 picks, the standard error of the headline is about 0.02, so differences below that are within noise. Ties in a score are broken by a seeded random draw.

## Baselines

The baselines are scored on canvass inspections in the validation window ([results/baselines.md](../../results/baselines.md)):

| Baseline | Weekly precision@k | Standard error |
|---|---|---|
| Logistic regression, v2 features | 0.449 | 0.023 |
| Logistic regression, v1 features | 0.394 | 0.022 |
| Category Fail rate of earlier fails (capped at 5) | 0.335 | 0.022 |
| Random ranking | 0.197 | 0.018 |

The validation base rate is 0.228. Rate baselines for ZIP code, previous result, and risk sit between random ranking and the logistic regressions.

### Rationale

- Single-column rate baselines show how much each signal is worth alone.
- The logistic regressions set the bar a trained model must clear, and the best of them becomes the promotion gate's threshold.

### Alternatives considered

- A base-rate baseline, which assigns every row the same score and so ranks at random.

## Code structure

Each stage is a plain function in a `src/` module (ingest, validate, clean, features, split, metrics, baselines), and `run_data.py` runs them in order. Settings live in `src/config.py`, with environment variable overrides. Unit tests cover the feature rules, leakage, split boundaries, the test guard, validation rules, and metrics.

### Rationale

- Plain functions run unchanged under DVC, under Prefect tasks, and in tests.
- One config module keeps every date, path, and threshold in one place.

### Alternatives considered

- Notebook cells that write modules to disk, which are hard to review, test, and rerun in order.

Assisting sources: Claude Opus 5.5 (Anthropic).
