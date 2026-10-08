import argparse
import json

import pytest

from src import dataset_acquire, dataset_review
from src.dataset_quota import apply_retirements, image_files


def fixture_row():
    return {"id": "one", "class": "apple", "source_identity": "source/one.jpg",
            "status": "pending_review", "original_path": "raw/one.jpg",
            "original_sha256": "original",
            "outputs": {"224": {"path": "224/one.png", "sha256": "large"},
                        "32": {"path": "32/one.png", "sha256": "small"}}}


def manifest(row):
    return {"schema_version": 1, "reason": "user_requested_quota_5000",
            "retired": [{"id": row["id"], "class": row["class"],
                         "source_identity": row["source_identity"], "files": image_files(row)}]}


def test_retirement_preserves_history_and_skips_missing_files(tmp_path):
    row = fixture_row()
    journal = tmp_path / "attempts.jsonl"
    journal.write_text(json.dumps(row) + "\n", encoding="utf-8")
    retirement = tmp_path / "quota_retirements.json"
    retirement.write_text(json.dumps(manifest(row)), encoding="utf-8")
    for reader in (dataset_acquire.load_records, dataset_review.snapshot_records):
        effective = reader(journal)
        assert effective[0]["status"] == "retired_quota"
        dataset_acquire.validate_existing(effective)
        assert json.loads(journal.read_text())["status"] == "pending_review"
        state = dataset_review.review_summary(effective, None)
        assert state["accepted_images"] == 0
        assert state["review_counts"]["apple"]["retired_quota"] == 1


@pytest.mark.parametrize("change", ["hash", "class", "duplicate", "unknown", "status"])
def test_retirement_rejects_bad_binding(tmp_path, change):
    row = fixture_row()
    value = manifest(row)
    item = value["retired"][0]
    if change == "hash":
        item["files"][0]["sha256"] = "tampered"
    elif change == "class":
        item["class"] = "orange"
    elif change == "duplicate":
        value["retired"].append(item)
    elif change == "unknown":
        item["id"] = "unknown"
    else:
        row["status"] = "failed"
    path = tmp_path / "quota_retirements.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        apply_retirements([row], path)


def test_no_retirement_leaves_rows_unchanged(tmp_path):
    rows = [fixture_row()]
    assert apply_retirements(rows, tmp_path / "absent.json") is rows


def test_resume_keeps_retired_ids_and_hashes_out_of_quota(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_acquire, "ROOT", tmp_path)
    monkeypatch.setattr(dataset_acquire, "META", tmp_path)
    monkeypatch.setattr(dataset_acquire, "validate_existing", lambda records: None)
    monkeypatch.setattr(dataset_acquire.time, "sleep", lambda _: None)
    monkeypatch.setattr(dataset_acquire.shutil, "disk_usage", lambda _: argparse.Namespace(free=10_000_000_000))
    retired = dict(fixture_row(), dataset_ref="source", native_pixel_sha256="native", pixel224_sha256="pixels")
    active = dict(retired, id="active", source_identity="source/active.jpg",
                  original_sha256="other", native_pixel_sha256="othernative", pixel224_sha256="otherpixels")
    (tmp_path / "attempts.jsonl").write_text(json.dumps(retired) + "\n" + json.dumps(active) + "\n", encoding="utf-8")
    (tmp_path / "quota_retirements.json").write_text(json.dumps(manifest(retired)), encoding="utf-8")
    alias = dict(retired, id="alias", source_identity="source/alias.jpg")
    fresh = dict(active, id="fresh", source_identity="source/fresh.jpg",
                 original_sha256="fresh", native_pixel_sha256="freshnative", pixel224_sha256="freshpixels")
    monkeypatch.setattr(dataset_acquire, "candidates", lambda: ([retired, active, alias, fresh], []))
    calls = []
    def fetch(row, local):
        calls.append(row["id"])
        return row.copy()
    monkeypatch.setattr(dataset_acquire, "fetch_candidate", fetch)
    args = argparse.Namespace(workers=4, hours=1, max_attempts=10, per_class=2, max_transfer_gb=1)
    state = dataset_acquire.run(args)
    assert state["counts"]["apple"]["pending_review"] == 2
    assert state["counts"]["apple"]["retired_quota"] == 1
    assert "one" not in calls and "active" not in calls
    # Depending on deterministic order, quota may stop before trying the alias.
    if "alias" in calls:
        assert state["counts"]["apple"]["duplicate_exact"] == 1
    before = calls.copy()
    dataset_acquire.run(args)
    assert calls == before