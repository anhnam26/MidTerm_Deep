"""User-confirmed acquisition review sidecar; never edit a live worker journal."""

import argparse
import collections
import json
import time

from .common import CLASSES, ROOT, write_json
from .dataset_quota import apply_retirements

META = ROOT / "data" / "metadata" / "fruit_acquisition_08"
BIND_FIELDS = ("id", "class", "source_identity", "original_sha256",
               "native_pixel_sha256", "pixel224_sha256")


def snapshot_records(path):
    """Read complete lines only; do not repair a worker-owned partial append."""
    rows = [json.loads(line) for line in path.read_bytes().splitlines(keepends=True)
            if line.endswith(b"\n")]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate IDs in acquisition journal")
    return apply_retirements(rows, path.parent / "quota_retirements.json")


def binding(row):
    value = {field: row[field] for field in BIND_FIELDS}
    if value["class"] not in CLASSES or not all(value.values()):
        raise ValueError("Invalid approval identity")
    value["output_sha256"] = {size: row["outputs"][size]["sha256"] for size in ("224", "32")}
    return value


def load_approval(path):
    if not path.exists():
        return None
    approval = json.loads(path.read_text(encoding="utf-8"))
    if (approval.get("schema_version") != 1 or approval.get("basis") != "user_confirmation"
            or approval.get("scope") != "completed_journal_snapshot"
            or not approval.get("statement", "").strip()):
        raise ValueError("Invalid user approval metadata")
    if len({item["id"] for item in approval["accepted"]}) != len(approval["accepted"]):
        raise ValueError("Duplicate approval IDs")
    return approval


def review_summary(records, approval):
    approved = {item["id"]: item for item in approval["accepted"]} if approval else {}
    counts = {fruit: collections.Counter() for fruit in CLASSES}
    development = collections.Counter()
    for row in records:
        status = row["status"]
        if status == "pending_review" and row["id"] in approved:
            if binding(row) != approved[row["id"]]:
                raise ValueError(f"Approval identity/hash mismatch: {row['id']}")
            status = "accepted_user_confirmed"
            if row.get("development_only"):
                development[row["class"]] += 1
        counts[row["class"]][status] += 1
    accepted = {c: counts[c]["accepted_user_confirmed"] for c in CLASSES}
    pending = sum(counts[c]["pending_review"] for c in CLASSES)
    return {"updated_unix": time.time(), "snapshot_attempts": len(records),
            "review_counts": {c: dict(counts[c]) for c in CLASSES},
            "accepted_images": sum(accepted.values()), "accepted_per_class": accepted,
            "accepted_shortfall_to_5050": {c: max(0, 5050 - accepted[c]) for c in CLASSES},
            "accepted_development_only_per_class": {c: development[c] for c in CLASSES},
            "pending_images": pending, "approval_basis": "user_confirmation" if approval else None,
            "approval_recorded_unix": approval["recorded_unix"] if approval else None,
            "content_rights_review_complete": approval is not None and pending == 0,
            "physical_groups_verified": False, "near_duplicates_reviewed": False,
            "review_complete": False, "official_export": False}


def record_approval(meta, statement):
    """Bind explicit user confirmation to already completed, technically valid rows."""
    from . import dataset_acquire as acquire

    if not statement.strip():
        raise ValueError("An explicit user confirmation statement is required")
    path = meta / "user_approval.json"
    if path.exists():
        raise FileExistsError("Approval already exists; do not overwrite review history")
    rows = snapshot_records(meta / "attempts.jsonl")
    acquire.validate_existing(rows)
    eligible = [row for row in rows if row["status"] == "pending_review"]
    if not eligible:
        raise ValueError("No technically valid pending images to approve")
    for row in eligible:
        if len(row["native_size"]) != 2 or min(row["native_size"]) < 224:
            raise ValueError(f"Invalid native geometry: {row['id']}")
    approval = {"schema_version": 1, "basis": "user_confirmation",
                "scope": "completed_journal_snapshot", "statement": statement,
                "recorded_unix": time.time(), "snapshot_attempts": len(rows),
                "independently_verified_by_agent": False,
                "accepted": [binding(row) for row in eligible]}
    # No journal/progress/lock/reservation writes. Raw review flags remain historical.
    write_json(path, approval)
    result = review_summary(rows, approval)
    write_json(meta / "review_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    accept = sub.add_parser("accept-snapshot")
    accept.add_argument("--statement", required=True, help="Actual explicit user confirmation")
    sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "accept-snapshot":
        result = record_approval(META, args.statement)
    else:
        rows = snapshot_records(META / "attempts.jsonl")
        result = review_summary(rows, load_approval(META / "user_approval.json"))
        write_json(META / "review_summary.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()