"""Tests for the prediction API in serving/app.py: request schema, behaviour, and a golden dataset.

The API runs in-process through FastAPI's TestClient, which starts the app (and loads the model)
without a server. The tests need the exported model in serving/model/ (see export_model.py), so
they are skipped where it is missing, for example in continuous integration.

- Schema tests: valid requests score, malformed requests are rejected with HTTP 422.
- Behavioural tests (Lecture 7): invariance (inputs the model should ignore do not change a score)
  and directional expectations (more risk signals raise a score).
- Golden dataset: fixed validation rows whose scores were recorded from the production model;
  any change in those scores means the model or its preprocessing changed.
"""

import importlib.util
import json

import pytest

from src import config

pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

GOLDEN = config.PROJECT_ROOT / "tests" / "data" / "golden_predictions.json"

pytestmark = pytest.mark.skipif(not (config.SERVING_MODEL_DIR / "version.json").exists(),
                                reason="no exported model in serving/model/")

FACILITY = {
    "facility_id": "A", "inspection_group": "canvass", "facility_group": "restaurant", "risk": "high",
    "prev_result": "Pass", "prior_fails": 2, "n_prior": 6, "days_since_prev": 365, "prev_n_violations": 4,
    "zip_fail_rate_365d": 0.22, "zip_complaints_90d": 10,
}


@pytest.fixture(scope="module")
def client():
    """A test client for serving/app.py; the with-block runs the app's startup, which loads the model."""
    spec = importlib.util.spec_from_file_location("serving_app", config.PROJECT_ROOT / "serving" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with TestClient(module.app) as test_client:
        yield test_client


def predict(client, *facilities):
    """POST the facilities to /predict and return the response."""
    return client.post("/predict", json={"facilities": list(facilities)})


def score(client, **changes) -> float:
    """Score of the base facility with `changes` applied."""
    return predict(client, {**FACILITY, **changes}).json()["results"][0]["score"]


# ---- schema ------------------------------------------------------------------------------------

def test_health_reports_the_model_version(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["semver"] and body["features"]


def test_predict_ranks_every_facility_highest_risk_first(client):
    low = {**FACILITY, "facility_id": "B", "prior_fails": 0, "prev_n_violations": 0, "zip_fail_rate_365d": 0.10}
    body = predict(client, low, FACILITY).json()
    assert [r["facility_id"] for r in body["results"]] == ["A", "B"]
    assert [r["rank"] for r in body["results"]] == [1, 2]
    assert all(0 < r["score"] < 1 for r in body["results"])
    assert body["model_version"] == client.get("/health").json()["semver"]


def test_first_inspection_without_history_is_scored(client):
    first = {**FACILITY, "prev_result": "first inspection", "prior_fails": 0, "n_prior": 0,
             "days_since_prev": None, "prev_n_violations": None}
    assert predict(client, first).status_code == 200


@pytest.mark.parametrize("bad_facility", [
    {**FACILITY, "dba_name": "TEST KITCHEN"},                              # field the model does not use
    {k: v for k, v in FACILITY.items() if k != "prior_fails"},             # required field missing
    {**FACILITY, "prior_fails": "three"},                                  # wrong type
])
def test_malformed_facilities_are_rejected(client, bad_facility):
    assert predict(client, bad_facility).status_code == 422


def test_empty_batch_is_rejected(client):
    assert client.post("/predict", json={"facilities": []}).status_code == 422


# ---- behaviour ---------------------------------------------------------------------------------

def test_score_ignores_the_client_id(client):
    """Invariance: the facility ID is a label for the client, not a model input."""
    assert score(client, facility_id="A") == score(client, facility_id="SOME-OTHER-ID")


def test_score_does_not_depend_on_the_rest_of_the_batch(client):
    """Invariance: a facility's score is the same alone and in a batch; only its rank is relative."""
    others = [{**FACILITY, "facility_id": f"X{i}", "prior_fails": i} for i in range(5)]
    alone = predict(client, FACILITY).json()["results"][0]["score"]
    batch = {r["facility_id"]: r["score"] for r in predict(client, FACILITY, *others).json()["results"]}
    assert batch["A"] == pytest.approx(alone)


@pytest.mark.parametrize("feature, low, high", [
    ("prior_fails", 0, 6),
    ("prev_n_violations", 0, 12),
    ("zip_fail_rate_365d", 0.10, 0.35),
])
def test_more_risk_signals_raise_the_score(client, feature, low, high):
    """Directional expectation: more earlier fails, violations, or ZIP code fails mean a higher score."""
    assert score(client, **{feature: high}) > score(client, **{feature: low})


# ---- golden dataset ----------------------------------------------------------------------------

def test_golden_dataset_scores_are_unchanged(client):
    """Regression test: the recorded validation rows still receive the recorded scores."""
    golden = json.loads(GOLDEN.read_text())
    assert golden["semver"] == client.get("/health").json()["semver"]
    results = {r["facility_id"]: r["score"] for r in predict(client, *golden["facilities"]).json()["results"]}
    for facility_id, expected in golden["expected_scores"].items():
        assert results[facility_id] == pytest.approx(expected, abs=1e-6)
