"""Tests for src/audit.py: audit records carry the request ID and the model lineage, not the data."""

import json

from src import audit

VERSION_INFO = {"model_name": "inspection-ranker", "registry_version": 1, "semver": "1.0.0",
                "data_tag": "data-v2", "git_commit": "abc123"}


def test_audit_records_carry_lineage_but_no_feature_values(tmp_path):
    request = [{"facility_id": "1", "risk": "high", "prior_fails": 2}]
    lines = [{"time": "t1", "scenario": "s", "week": "w", "status": 200, "request": request,
              "response": {"model_version": "1.0.0", "results": []}},
             {"time": "t2", "scenario": "s", "week": "w", "status": 422, "request": request,
              "response": {"detail": []}}]
    log = tmp_path / "requests.jsonl"
    log.write_text("".join(json.dumps(line) + "\n" for line in lines))
    accepted, rejected = audit.audit_records(log, VERSION_INFO)
    assert accepted["request_id"] == rejected["request_id"] == audit.request_id(request)
    assert accepted["model_version"] == "1.0.0" and rejected["model_version"] is None
    assert accepted["data_version"] == "data-v2" and "risk" not in accepted
