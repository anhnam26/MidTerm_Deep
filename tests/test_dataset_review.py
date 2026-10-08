import copy
import hashlib
import json

from PIL import Image
import pytest

from src import dataset_acquire as acquire, dataset_review as review


def candidate(identifier="one", status="pending_review"):
    return {"id": identifier, "class": "apple", "status": status,
            "source_identity": "owner/source/" + identifier,
            "original_sha256": identifier, "native_pixel_sha256": identifier,
            "pixel224_sha256": identifier, "native_size": [300, 240],
            "outputs": {"224": {"sha256": "224"}, "32": {"sha256": "32"}},
            "photo_verified": False, "rights_verified": False,
            "physical_groups_verified": False, "development_only": True}


def approval(rows):
    return {"schema_version": 1, "basis": "user_confirmation",
            "scope": "completed_journal_snapshot", "statement": "User confirms valid images and rights",
            "recorded_unix": 1, "accepted": [review.binding(row) for row in rows]}


def test_snapshot_does_not_repair_live_journal(tmp_path):
    path = tmp_path / "attempts.jsonl"
    content = json.dumps(candidate()).encode() + b'\n{"id":'
    path.write_bytes(content)
    assert review.snapshot_records(path) == [candidate()]
    assert path.read_bytes() == content
    path.write_bytes(b'invalid\n')
    with pytest.raises(ValueError):
        review.snapshot_records(path)


def test_snapshot_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "attempts.jsonl"
    path.write_text((json.dumps(candidate()) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate IDs"):
        review.snapshot_records(path)


def test_no_approval_and_snapshot_only_acceptance():
    first, later = candidate(), candidate("later")
    failed, duplicate = candidate("failed", "failed"), candidate("dup", "duplicate_exact")
    rows = [first, later, failed, duplicate]
    before = copy.deepcopy(rows)
    assert review.review_summary(rows, None)["accepted_images"] == 0
    result = review.review_summary(rows, approval([first]))
    assert result["accepted_images"] == 1 and result["pending_images"] == 1
    assert result["accepted_development_only_per_class"]["apple"] == 1
    assert result["accepted_shortfall_to_5050"]["apple"] == 5049
    assert result["review_counts"]["apple"] == {
        "accepted_user_confirmed": 1, "pending_review": 1, "failed": 1, "duplicate_exact": 1}
    assert not result["review_complete"] and not result["physical_groups_verified"]
    assert not result["near_duplicates_reviewed"] and rows == before


@pytest.mark.parametrize("field", ["original_sha256", "source_identity", "class", "outputs"])
def test_approval_bound_to_identity_and_hashes(field):
    row = candidate()
    recorded = approval([row])
    row[field] = {"224": {"sha256": "changed"}, "32": {"sha256": "32"}} if field == "outputs" else "banana"
    with pytest.raises(ValueError, match="mismatch"):
        review.review_summary([row], recorded)


def test_approval_never_revives_failed_or_duplicate():
    row = candidate()
    recorded = approval([row])
    for status in ("failed", "duplicate_exact"):
        row["status"] = status
        assert review.review_summary([row], recorded)["accepted_images"] == 0


def test_invalid_metadata(tmp_path):
    path = tmp_path / "user_approval.json"
    assert review.load_approval(path) is None
    value = approval([candidate()])
    value["basis"] = "agent_verified"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Invalid"):
        review.load_approval(path)
    value = approval([candidate(), candidate()])
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Duplicate approval"):
        review.load_approval(path)


def test_record_snapshot_preserves_worker_files_and_validates_content(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire, "ROOT", tmp_path)
    row = candidate()
    for name, size in (("original", 300), ("224", 224), ("32", 32)):
        path = tmp_path / (name + ".png")
        Image.new("RGB", (size, size), "red").save(path)
        item = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if name == "original":
            row.update(original_path=path.name, original_sha256=item["sha256"])
        else:
            row["outputs"][name] = item
    journal = tmp_path / "attempts.jsonl"
    journal.write_text(json.dumps(row) + "\n" + json.dumps(candidate("failed", "failed")) + "\n")
    for name in ("progress.json", "worker.lock", "development_reservations.json"):
        (tmp_path / name).write_bytes(b"preserve")
    before = journal.read_bytes()
    # Approval cannot bypass persisted hashes.
    (tmp_path / "32.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Changed/missing"):
        review.record_approval(tmp_path, "User confirms")
    assert not (tmp_path / "user_approval.json").exists()
    Image.new("RGB", (32, 32), "red").save(tmp_path / "32.png")
    result = review.record_approval(tmp_path, "User confirms")
    assert result["accepted_images"] == 1 and result["content_rights_review_complete"]
    assert not review.load_approval(tmp_path / "user_approval.json")["independently_verified_by_agent"]
    assert journal.read_bytes() == before
    for name in ("progress.json", "worker.lock", "development_reservations.json"):
        assert (tmp_path / name).read_bytes() == b"preserve"
    with pytest.raises(FileExistsError):
        review.record_approval(tmp_path, "User confirms")


def test_worker_summary_keeps_raw_resume_status(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire, "META", tmp_path)
    row = candidate()
    (tmp_path / "user_approval.json").write_text(json.dumps(approval([row])))
    state = acquire.summary([row], [row], "running", set())
    assert state["accepted_images"] == 1
    assert state["counts"]["apple"] == {"pending_review": 1}
    assert row["status"] == "pending_review"
    assert state["pending_shortfall_to_5050"]["apple"] == 5049
    assert state["accepted_shortfall_to_5050"]["apple"] == 5049
    assert state["approval_basis"] == "user_confirmation"