# Team setup

This guide gets a team member from a fresh checkout to a working environment with the project's data, in about 15 minutes. Every command runs from the repository root. The [README](../README.md) explains the pipeline, the data versions, and the results in full.

## 1. Create the environment

Use a fresh virtual environment with Python 3.10 or later. A conda `base` environment can hold NumPy and pandas builds that conflict and stop pandas from importing.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip check
```

On Windows, activate the environment with `.venv\Scripts\activate`. `pip check` should print "No broken requirements found."

`requirements.txt` pins `cryptography` and `pyOpenSSL` to versions that DVC's Google Drive client works with. An environment created before those pins needs `pip install -r requirements.txt` again.

## 2. Connect to the DVC remote

The data lives in a folder in the team's Google Shared Drive, which `.dvc/config` already names. DVC reaches it through a Google Cloud service account, so each team member needs its key file.

1. Get the service account's JSON key file from the repository owner through a private channel. Never commit it, email it to a public list, or paste it into an issue.
2. Save it outside the repository, for example in `~/.config/dvc/`, and limit it to your user:

```bash
mkdir -p ~/.config/dvc
chmod 600 ~/.config/dvc/<key-file>.json
```

3. Point DVC at it. These settings go to `.dvc/config.local`, which git ignores:

```bash
dvc remote modify --local gdrive gdrive_use_service_account true
dvc remote modify --local gdrive gdrive_service_account_json_file_path ~/.config/dvc/<key-file>.json
```

## 3. Get the data

```bash
dvc pull
```

This downloads the Food Inspections CSV, the raw snapshot (`data/raw/`), and the splits (`data/processed/`) for the checked-out commit.

## 4. Check the setup

```bash
python -m pytest
dvc status
```

All tests should pass, and `dvc status` should report that the data and pipelines are up to date.

## 5. Run the training pipeline

```bash
python pipeline.py --time-budget 60 --no-register
```

This quick run checks that Prefect, FLAML, and MLflow work, without registering a model. The full run, `python pipeline.py`, searches for 300 seconds per feature version and registers the winner if it passes the gate. View the runs with `mlflow ui --backend-store-uri sqlite:///mlflow.db` at http://127.0.0.1:5000. Each person's `mlflow.db` is local, so registered models stay on the machine that trained them.

## Working with data versions

Each data version is a git tag. Both load through `load_split()`, so the code is the same for either.

| Tag | Facility scope |
|---|---|
| `data-v1` | all facility types |
| `data-v2` | restaurants only, the version the model trains on |

To switch versions, check out the tag and let DVC restore the matching data:

```bash
git checkout data-v1
dvc checkout
```

Return to your branch the same way, with `git checkout <branch>` followed by `dvc checkout`. To compare the baselines of the two versions, run `dvc metrics diff data-v1 data-v2`.

## Changing the data

Settings that shape the data, such as `FACILITY_SCOPE`, live in `src/config.py`. After changing a setting or any code the pipeline depends on:

1. Run `dvc repro`, which reruns only the stages whose inputs changed and updates `dvc.lock`.
2. Commit `dvc.lock`, the code, and the updated files in `results/` on a feature branch.
3. Run `dvc push`, so teammates can pull the new data.
4. For a new data version, tag the commit (`git tag -a data-v3 -m "..."`) and push the tag with `git push origin data-v3`.

Changes reach `main` through a pull request with one reviewer.

## Without the remote

The data is public, so the pipeline can rebuild it without `dvc pull`. Save the export from the [Food Inspections dataset page](https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5) as `data/Food_Inspections_20261102.csv` and run `dvc repro`, which takes about a minute. The rows match the shared data, while `dvc status` may still report differences from file metadata such as the raw manifest's timestamp.

## Troubleshooting

| Message | Cause and fix |
|---|---|
| `module 'lib' has no attribute 'GEN_EMAIL'` or `'X509_V_FLAG_NOTIFY_POLICY'` | `pyOpenSSL` and `cryptography` versions do not match. Run `pip install -r requirements.txt`, then `pip check`. |
| `numpy.core.multiarray failed to import` | The command ran in conda `base`. Activate `.venv` and rerun it. |
| `WARNING: failed to collect 'mpiira-data-v1', skipping` | That tag predates the DVC setup at the repository root and has no data to push or pull. The warning is harmless. |
| A permissions or file-not-found error from Google Drive | The service account is missing from the Shared Drive, or `.dvc/config.local` points to the wrong key path. Check the path with `dvc config --local --list`. |
| `ERROR: failed to pull data from the cloud` with missing files | The data for that commit was never pushed. Ask whoever made the commit to run `dvc push`. |
