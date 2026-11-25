# Reproducibility checklist

| Item | How the project handles it | Where |
|---|---|---|
| Code version | every MLflow run and the exported model record the git commit, and whether the code had uncommitted changes | `src/tracking.py`, `serving/model/version.json` |
| Data version | DVC tags `data-v1` and `data-v2` restore each data version; `dvc.lock` holds the hash of every stage's inputs and outputs | `dvc.yaml`, `dvc.lock` |
| Split integrity | each split's content hash is in the split manifest and is checked every time `load_split` reads it; the test split needs `allow_test=True` | `src/split.py`, `data/processed/split_manifest.json` |
| Random seeds | one seed (`config.SEED = 42`) for the AutoML search, tie-breaking in the metric, sampling and the drift scenarios | `src/config.py` |
| Fixed dates | the snapshot end, the split dates and the food code change date are constants | `src/config.py` |
| Model lineage | the registry version carries a semantic version tag and lineage tags (data tag, DVC fingerprint of the splits, git commit) | `src/registry.py`, MLflow |
| Dependencies | the serving image pins exact versions; the project's `requirements.txt` sets minimum versions, with upper limits where a library breaks (pandas below 3, xgboost below 3, Evidently below 0.8) | `serving/requirements.txt`, `requirements.txt` |
| Environment | the API runs in a Docker image from a pinned base image (`python:3.13-slim`); tests run on Python 3.12 in continuous integration | `Dockerfile`, `.github/workflows/tests.yml` |
| Regression checks | unit tests on every push, and a golden dataset that checks the production model's scores are unchanged | `tests/` |
| End-to-end rebuild | without an exported model, `app/app.ipynb` re-runs the training pipeline into `rebuild/`, which registers the same model (1.0.0, validation weekly precision@k 0.449) | `app/app.ipynb`, Session 5 |
