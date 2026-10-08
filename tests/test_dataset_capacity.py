import hashlib
import json

import pytest
from PIL import Image

from src import data
from src.common import write_csv


def test_decimal_capacity_limits():
    assert data.DATASET_LIMIT_BYTES == {224: 25_000_000, 32: 20_000_000}


@pytest.mark.parametrize("size,old_limit", [(224, 26_214_400), (32, 20_971_520)])
def test_old_mib_limits_are_rejected(size, old_limit):
    with pytest.raises(ValueError, match="Total dataset capacity limit exceeded"):
        data.capacity_report(size, old_limit)


@pytest.mark.parametrize("size", [224, 32])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_capacity_boundary(size, offset):
    limit = data.DATASET_LIMIT_BYTES[size]
    if offset > 0:
        with pytest.raises(ValueError, match=f"Total dataset capacity limit exceeded for Mid{size}"):
            data.capacity_report(size, limit + offset)
    else:
        report = data.capacity_report(size, limit + offset)
        assert report["total_bytes"] == limit + offset
        assert report["remaining_bytes"] == -offset
        assert report["total_mb"] == (limit + offset) / 1_000_000
        assert report["limit_mb"] == (25 if size == 224 else 20)
        assert report["total_mib"] == (limit + offset) / 1024 ** 2
        assert report["limit_mib"] == limit / 1024 ** 2


@pytest.fixture
def paired_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA", tmp_path / "data")
    monkeypatch.setattr(data, "META", data.DATA / "metadata")
    monkeypatch.setattr(data, "CLASSES", ("apple",))
    monkeypatch.setattr(data, "SPLIT_QUOTAS", {"train": 1, "val": 1, "test": 1})
    rows = []
    for index, split in enumerate(data.SPLIT_QUOTAS):
        digest = hashlib.sha256(split.encode()).hexdigest()
        row = {"id": split, "class": "apple", "split": split, "group_id": split,
               "original_sha256": digest, "sha256": digest, "dhash": digest[:16]}
        for size in data.DATASET_LIMIT_BYTES:
            path = data.DATA / f"Mid{size}" / split / "apple" / f"{split}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (size, size), (index * 80, 50, 120)).save(path)
            row[f"mid{size}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(row)
    write_csv(data.META / "splits.csv", rows)
    totals = {size: sum(path.stat().st_size for path in (data.DATA / f"Mid{size}").rglob("*.png"))
              for size in data.DATASET_LIMIT_BYTES}
    return rows, totals


@pytest.mark.parametrize("size", [224, 32])
@pytest.mark.parametrize("margin", [-1, 0, 1])
def test_audit_checks_combined_splits(paired_dataset, monkeypatch, size, margin):
    rows, totals = paired_dataset
    limits = dict(data.DATASET_LIMIT_BYTES)
    limits[size] = totals[size] + margin
    monkeypatch.setattr(data, "DATASET_LIMIT_BYTES", limits)
    # Every individual file is below the limit, even when their sum exceeds it.
    assert all(path.stat().st_size < limits[size] for path in (data.DATA / f"Mid{size}").rglob("*.png"))
    if margin < 0:
        with pytest.raises(ValueError, match=f"Total dataset capacity limit exceeded for Mid{size}"):
            data.audit()
        assert not (data.META / "audit.json").exists()
    else:
        report = data.audit()
        assert report[f"Mid{size}"]["images"] == len(rows)
        assert report[f"Mid{size}"]["total_bytes"] == totals[size]
        assert report[f"Mid{size}"]["remaining_bytes"] == margin
        assert report[f"Mid{size}"]["counts"] == {split: {"apple": 1} for split in data.SPLIT_QUOTAS}
        assert json.loads((data.META / "audit.json").read_text()) == report


def test_audit_rejects_non_png_extra_file(paired_dataset):
    extra = data.DATA / "Mid224" / "extra.jpg"
    Image.new("RGB", (224, 224)).save(extra)
    with pytest.raises(ValueError, match="Unmanifested/missing"):
        data.audit()


def test_audit_retains_hash_check(paired_dataset):
    Image.new("RGB", (32, 32), "green").save(data.DATA / "Mid32" / "test" / "apple" / "test.png")
    with pytest.raises(ValueError, match="Image content changed"):
        data.audit()


@pytest.mark.parametrize("size", [224, 32])
@pytest.mark.parametrize("dimensions", ["wrong_width", "wrong_height", "other_resolution"])
def test_audit_rejects_wrong_dimensions(paired_dataset, size, dimensions):
    shape = {"wrong_width": (size - 1, size), "wrong_height": (size, size - 1),
             "other_resolution": (32, 32) if size == 224 else (224, 224)}[dimensions]
    path = data.DATA / f"Mid{size}" / "train" / "apple" / "train.png"
    Image.new("RGB", shape).save(path)
    with pytest.raises(ValueError, match="Invalid mode/size"):
        data.audit()


def test_audit_retains_quota_check(paired_dataset, monkeypatch):
    monkeypatch.setattr(data, "SPLIT_QUOTAS", {"train": 2, "val": 1, "test": 1})
    with pytest.raises(ValueError, match="Wrong quota"):
        data.audit()


@pytest.mark.parametrize("failure", [224, 32, "quota"])
def test_failed_build_never_freezes_or_exports(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(data, "DATA", tmp_path / "data")
    monkeypatch.setattr(data, "META", data.DATA / "metadata")
    monkeypatch.setattr(data, "CLASSES", ("apple",))
    monkeypatch.setattr(data, "SPLIT_QUOTAS", {"train": 2 if failure == "quota" else 1, "val": 0, "test": 0})
    crop = tmp_path / "crop.png"
    Image.new("RGB", (300, 240), "red").save(crop)
    digest = hashlib.sha256(crop.read_bytes()).hexdigest()
    row = {"id": "fixture", "class": "apple", "split": "train", "group_id": "fixture",
           "original_sha256": digest, "sha256": digest, "dhash": "0" * 16,
           "crop_width": "300", "crop_height": "240", "crop_path": "crop.png", "status": "pending_review"}
    write_csv(data.META / "downloads.csv", [row])
    write_csv(data.META / "reviews.csv", [{"id": "fixture", "decision": "accept"}])
    monkeypatch.setattr(data, "select_splits", lambda *args, **kwargs: [dict(row)])
    if failure != "quota":
        limits = dict(data.DATASET_LIMIT_BYTES)
        limits[failure] = 1
        monkeypatch.setattr(data, "DATASET_LIMIT_BYTES", limits)
    with pytest.raises(ValueError, match="Wrong quota" if failure == "quota" else "Total dataset capacity"):
        data.build(2026)
    assert not (data.META / "splits.csv").exists()
    assert not (data.META / "build_settings.json").exists()
    assert not (data.META / "audit.json").exists()
    assert not (data.DATA / "Mid224").exists()
    assert not (data.DATA / "Mid32").exists()
    assert not list(data.DATA.glob("paired_build_*"))
    assert crop.exists()
    assert (data.META / "downloads.csv").exists()
    assert (data.META / "reviews.csv").exists()


def test_build_preserves_existing_dataset_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA", tmp_path)
    monkeypatch.setattr(data, "META", tmp_path / "metadata")
    existing = tmp_path / "Mid32"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("existing data")
    with pytest.raises(ValueError, match="No overwrite allowed"):
        data.build(2026)
    assert marker.read_text() == "existing data"