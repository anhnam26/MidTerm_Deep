import argparse
import hashlib
import io
import json
import urllib.error
import zipfile

from PIL import Image
import pytest

from src import dataset_acquire as acquire


def row(identifier="one", fruit="apple", ref="owner/source"):
    return {"id": identifier, "class": fruit, "dataset_ref": ref,
            "source_name": identifier + ".jpg", "download_url": "https://example.invalid/image",
            "group_id": "test_group"}


def blob():
    stream = io.BytesIO()
    Image.new("RGB", (300, 240), "red").save(stream, format="JPEG")
    return stream.getvalue()


class Response(io.BytesIO):
    headers = {}


def test_folder_labels_and_derivative_exclusions():
    assert acquire.identify("train/Apple/x.JPG") == "apple"
    assert acquire.identify("Good/Orange_Good/x.jpg") == "orange"
    assert acquire.identify("Total Number of Apples/x.png") is None
    assert acquire.identify("train/apple/x_edited.jpg") is None
    assert acquire.identify("apple/example.txt") is None
    assert acquire.identify("pineapple_background/x.jpg") is None
    assert acquire.identify("all/images/x.jpg", "strawberry") == "strawberry"


def test_resize_direct_pairs_and_no_upscale():
    native, pairs = acquire.prepare(Image.new("RGB", (300, 240), "red"))
    assert native.size == (300, 240)
    assert pairs[224].size == (224, 224) and pairs[32].size == (32, 32)
    assert pairs[224].getpixel((0, 0)) == (127, 127, 127)
    with pytest.raises(ValueError, match="upscaling"):
        acquire.prepare(Image.new("RGB", (223, 300)))


def test_official_openimages_requires_stable_identity_license_and_provenance():
    source = dict(id="oi_0123456789abcdef", provider="openimages", source_id="0123456789abcdef",
        source_split="train", source_identity="openimages/0123456789abcdef",
        download_url="https://open-images-dataset.s3.amazonaws.com/train/0123456789abcdef.jpg",
        license_declared="https://creativecommons.org/licenses/by/2.0/", author="Photographer",
        landing_url="https://www.flickr.com/photos/author/1", source_url="https://staticflickr.com/photo.jpg",
        label_evidence="human_positive")
    assert acquire.permitted_candidate_url(source)
    for key, value in (("source_id", "../../secret"), ("source_identity", "alias"),
                       ("download_url", "https://example.com/photo.jpg"), ("source_split", "other"),
                       ("author", ""), ("license_declared", "Unknown"),
                       ("id", "new_alias"), ("label_evidence", "machine")):
        assert not acquire.permitted_candidate_url(dict(source, **{key: value}))
    assert acquire.permitted_candidate_url({"download_url": "https://www.kaggle.com/api/v1/datasets/download/owner/source/photo.jpg"})


def test_fetch_local_reuse_and_pixel_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire, "ROOT", tmp_path)
    monkeypatch.setattr(acquire, "OUT", tmp_path / "staging")
    monkeypatch.setattr(acquire.data, "development_reservations", lambda: {})
    calls = []
    def fetch(*args, **kwargs):
        calls.append(args)
        return Response(blob())
    monkeypatch.setattr(acquire.urllib.request, "urlopen", fetch)
    first = acquire.fetch_candidate(row())
    assert first["status"] == "pending_review" and len(calls) == 1
    second = acquire.fetch_candidate(row())
    assert second["status"] == "pending_review" and len(calls) == 1
    assert second["reused_local"] and second["transferred_bytes"] == 0
    hashes = {}
    acquire.mark_duplicate(first, hashes)
    acquire.mark_duplicate(second, hashes)
    assert second["status"] == "duplicate_exact"
    for size, item in first["outputs"].items():
        with Image.open(tmp_path / item["path"]) as opened:
            assert opened.size == (int(size), int(size)) and opened.mode == "RGB"
    acquire.validate_existing([first])
    (tmp_path / first["outputs"]["32"]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Changed/missing"):
        acquire.validate_existing([first])


def test_pilot_reuse_retains_reserved_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire, "ROOT", tmp_path)
    monkeypatch.setattr(acquire, "OUT", tmp_path / "staging")
    monkeypatch.setattr(acquire.data, "development_reservations", lambda: {"ids": {"one"}})
    path = tmp_path / "pilot.jpg"
    path.write_bytes(blob())
    local = {"one": {"local_path": "pilot.jpg", "original_sha256": hashlib.sha256(blob()).hexdigest()}}
    result = acquire.fetch_candidate(row(), local)
    assert result["reused_pilot"] and result["development_only"]
    assert result["original_path"] == "pilot.jpg" and result["transferred_bytes"] == 0


def test_bounded_download_and_http_no_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire, "OUT", tmp_path)
    monkeypatch.setattr(acquire, "MAX_FILE_BYTES", 3)
    with pytest.raises(ValueError, match="safety cap"):
        acquire.read_bounded(Response(b"1234"))
    calls = []
    def fetch(url, **kwargs):
        calls.append(url)
        raise urllib.error.HTTPError(url, 429, "rate limit", {}, None)
    monkeypatch.setattr(acquire.urllib.request, "urlopen", fetch)
    result = acquire.fetch_candidate(row())
    assert result["http_status"] == 429 and result["status"] == "failed" and len(calls) == 1


def test_registered_local_source_requires_expected_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire, "ROOT", tmp_path)
    monkeypatch.setattr(acquire, "OUT", tmp_path / "staging")
    monkeypatch.setattr(acquire.data, "development_reservations", lambda: {})
    original = acquire.OUT / "originals/apple/one.jpg"
    original.parent.mkdir(parents=True)
    original.write_bytes(blob())
    result = acquire.fetch_candidate(dict(row(), source_sha256="0" * 64))
    assert result["status"] == "failed" and "Registered source hash mismatch" in result["error"]
    result = acquire.fetch_candidate(dict(row(), source_sha256=hashlib.sha256(blob()).hexdigest()))
    assert result["status"] == "pending_review" and result["reused_local"]
    result = acquire.fetch_candidate(dict(row(), label_evidence="human_positive"))
    assert result["label_source"] == "human_positive"


def test_mendeley_archive_allowlist_and_member_identity():
    doi = "10.17632/tv7zfd6yzz.1"
    member = "PineappleLeaf-Fruit Dataset/Raw Dataset/Fruits/one.jpg"
    identity = doi + "/" + member
    source = dict(provider="mendeley_archive", source_doi=doi, archive_member=member, class_="pineapple",
        source_identity=identity, id="mendeley_" + hashlib.sha256(identity.encode()).hexdigest()[:20],
        download_url="https://data.mendeley.com/public-files/datasets/tv7zfd6yzz/files/8b6320ff-0d8f-47f5-8cfb-58a00d515b51/file_downloaded",
        source_sha256="a" * 64, archive_sha256="b" * 64, author="Authors",
        landing_url="https://data.mendeley.com/datasets/tv7zfd6yzz/1",
        license_declared="https://creativecommons.org/licenses/by/4.0/")
    source["class"] = source.pop("class_")
    assert acquire.permitted_candidate_url(source)
    for key, value in (("archive_member", "../unsafe.jpg"), ("source_doi", "unknown"),
                       ("id", "alias"), ("source_sha256", "short"), ("archive_sha256", ""),
                       ("archive_member", "PineappleLeaf-Fruit Dataset/Augmented Dataset/Fruits/aug_1.jpg"),
                       ("archive_member", "PineappleLeaf-Fruit Dataset/Raw Dataset/Healthy_leaf/one.jpg"),
                       ("class", "strawberry")):
        assert not acquire.permitted_candidate_url(dict(source, **{key: value}))


def test_journal_recovers_only_partial_tail(tmp_path):
    path = tmp_path / "attempts.jsonl"
    path.write_bytes(b'{"id":"a"}\n{"id":')
    assert acquire.load_records(path) == [{"id": "a"}]
    assert path.with_suffix(".interrupted_tail").exists()
    path.write_bytes(b'not json\n{"id":"a"}\n')
    with pytest.raises(ValueError, match="Corrupted"):
        acquire.load_records(path)


def test_single_file_zip_is_bounded_and_never_extracts_paths(monkeypatch):
    payload = blob()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("one.jpg", payload)
    assert acquire.unwrap_image(stream.getvalue(), "upstream/one.jpg") == payload
    assert acquire.unwrap_image(payload, "one.jpg") == payload
    with pytest.raises(ValueError, match="Unexpected"):
        acquire.unwrap_image(stream.getvalue(), "other.jpg")
    monkeypatch.setattr(acquire, "MAX_FILE_BYTES", len(payload) - 1)
    with pytest.raises(ValueError, match="oversized"):
        acquire.unwrap_image(stream.getvalue(), "one.jpg")
    with zipfile.ZipFile(stream, "a") as archive:
        archive.writestr("extra.jpg", b"extra")
    with pytest.raises(ValueError, match="Unexpected"):
        acquire.unwrap_image(stream.getvalue(), "one.jpg")


def test_run_resume_fairness_and_blocked_source(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire, "ROOT", tmp_path)
    monkeypatch.setattr(acquire, "META", tmp_path / "meta")
    monkeypatch.setattr(acquire.time, "sleep", lambda _: None)
    # Small temporary filesystem reports may not have5GB; only mocked in this test.
    monkeypatch.setattr(acquire.shutil, "disk_usage", lambda _: argparse.Namespace(free=10_000_000_000))
    choices = [row(str(i), c, c + "/source") for c in acquire.CLASSES for i in range(3)]
    # Real source IDs are globally unique; this fixture must be too.
    for r in choices:
        r["id"] = r["class"] + r["id"]
    monkeypatch.setattr(acquire, "candidates", lambda: (choices, []))
    calls = []
    def fetch(r, local):
        calls.append(r["class"])
        return dict(r, status="failed", http_status=429, transferred_bytes=0)
    monkeypatch.setattr(acquire, "fetch_candidate", fetch)
    args = argparse.Namespace(workers=1, hours=1, max_attempts=10, per_class=10, max_transfer_gb=1)
    state = acquire.run(args)
    assert sorted(calls) == sorted(acquire.CLASSES)
    assert len(state["blocked_sources"]) == 5 and not (acquire.META / "worker.lock").exists()
    acquire.run(args)
    assert len(calls) == 5  # No re-download or automatic retry of blocked sources.
    (acquire.META / "worker.lock").write_text(json.dumps({"pid": 123}))
    with pytest.raises(FileExistsError):
        acquire.run(args)


def test_single_source_fills_worker_batch_without_quota_overrun(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire, "ROOT", tmp_path)
    monkeypatch.setattr(acquire, "META", tmp_path / "meta")
    monkeypatch.setattr(acquire.time, "sleep", lambda _: None)
    monkeypatch.setattr(acquire.shutil, "disk_usage", lambda _: argparse.Namespace(free=10_000_000_000))
    monkeypatch.setattr(acquire, "validate_existing", lambda _: None)
    monkeypatch.setattr(acquire, "mark_duplicate", lambda *_: None)
    choices = [row(str(i), "strawberry") for i in range(10)]
    monkeypatch.setattr(acquire, "candidates", lambda: (choices, []))
    batches = []

    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def map(self, fn, items):
            items = list(items)
            batches.append(len(items))
            return map(fn, items)

    monkeypatch.setattr(acquire.concurrent.futures, "ThreadPoolExecutor", Pool)
    monkeypatch.setattr(acquire, "fetch_candidate", lambda r, _: dict(r, status="pending_review", transferred_bytes=0,
        original_sha256=r["id"], native_pixel_sha256=r["id"], pixel224_sha256=r["id"]))
    args = argparse.Namespace(workers=4, hours=1, max_attempts=10, per_class=5, max_transfer_gb=1)
    state = acquire.run(args)
    assert batches == [4, 1]
    assert state["counts"]["strawberry"]["pending_review"] == 5
    assert state["stop_reason"] == "target_or_inventory_exhausted"
    assert not (acquire.META / "worker.lock").exists()