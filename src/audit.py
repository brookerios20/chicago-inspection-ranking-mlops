"""Structured audit log: one JSON record per request batch the prediction API received.

Each record answers "which model, trained on which data and code, answered which request":
the request ID, the time, the HTTP status, the number of facilities, the model version the API
returned, and the lineage of the deployed model (registry version, data version tag, git commit).
The request ID is a hash of the request body, so the same request always has the same ID and a
record can be matched to the request log without copying any feature values into the audit log.
"""

import hashlib
import json
from pathlib import Path


def request_id(request: list[dict]) -> str:
    """A stable ID for a request: the first 16 hex digits of the SHA-256 hash of its JSON body."""
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:16]


def audit_records(request_log: Path, version_info: dict) -> list[dict]:
    """One audit record per batch in the monitoring request log, with the deployed model's lineage."""
    records = []
    for line in Path(request_log).read_text().splitlines():
        batch = json.loads(line)
        records.append({
            "time": batch["time"],
            "request_id": request_id(batch["request"]),
            "scenario": batch["scenario"],
            "week": batch["week"],
            "status": batch["status"],
            "facilities": len(batch["request"]),
            "model_version": batch["response"].get("model_version"),   # None when the request was rejected
            "model_name": version_info["model_name"],
            "registry_version": version_info["registry_version"],
            "data_version": version_info["data_tag"],
            "git_commit": version_info["git_commit"],
        })
    return records


def write_audit_log(records: list[dict], path: Path) -> None:
    """Write the records as JSON lines, one record per line."""
    Path(path).write_text("".join(json.dumps(record) + "\n" for record in records))
