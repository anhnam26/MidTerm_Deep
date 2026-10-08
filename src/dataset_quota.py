"""Apply explicit quota retirements without rewriting acquisition history."""

import json


def image_files(row):
    return [{"path": row["original_path"], "sha256": row["original_sha256"]},
            *[{"path": item["path"], "sha256": item["sha256"]}
              for item in row["outputs"].values()]]


def apply_retirements(records, path):
    if not path.exists():
        return records
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("reason") != "user_requested_quota_5000":
        raise ValueError("Invalid retirement manifest")
    retired = {item["id"]: item for item in manifest["retired"]}
    if len(retired) != len(manifest["retired"]):
        raise ValueError("Duplicate retirement IDs")
    known = {row["id"] for row in records}
    if not set(retired) <= known:
        raise ValueError("Retirement IDs missing from journal")
    result = []
    for row in records:
        item = retired.get(row["id"])
        if item:
            if (row["status"] != "pending_review" or row["class"] != item["class"]
                    or row["source_identity"] != item["source_identity"]
                    or image_files(row) != item["files"]):
                raise ValueError(f"Retirement binding mismatch: {row['id']}")
            row = dict(row, status="retired_quota")
        result.append(row)
    return result