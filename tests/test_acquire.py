from PIL import Image
import pytest

from src import acquire, data
from src.common import read_csv, write_csv


def test_positive_labels_no_negative_machine_or_mixed():
    def label(identifier, mid, confidence="1", source="verification"):
        return {"ImageID": identifier, "LabelName": mid, "Confidence": confidence, "Source": source}
    labels = [label("good", "/m/014j1m"), label("negative", "/m/014j1m", "0"),
              label("machine", "/m/014j1m", source="machine"), label("mixed", "/m/014j1m"), label("mixed", "/m/09qck")]
    assert acquire.positive_targets(labels) == {"good": "apple"}


def test_commons_license_size_and_author():
    page = {"pageid": 123, "title": "File:Fruit.jpg", "imageinfo": [{"width": 640, "height": 480, "mime": "image/jpeg",
            "url": "https://example.test/image.jpg", "descriptionurl": "https://example.test/page",
            "extmetadata": {"LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0/"},
                            "Artist": {"value": "<a href='link'>Name &amp; Co</a>"}}}]}
    row = acquire.commons_row(page, "pineapple")
    assert row["eligible"] == "1" and row["author"] == "Name & Co"
    page["imageinfo"][0]["extmetadata"]["LicenseUrl"]["value"] = "https://example.test/all-rights-reserved"
    assert acquire.commons_row(page, "pineapple")["eligible"] == "0"


def test_inventory_excludes_cross_category_conflicts(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "META", tmp_path)
    for fruit in ("apple", "banana"):
        write_csv(tmp_path / f"expanded_{fruit}.csv", [{"id": "wc_1", "class": fruit, "eligible": "1"}], acquire.FIELDS)
    assert acquire.candidates() == []


def test_download_resume_and_existing_crop_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire, "ROOT", tmp_path)
    monkeypatch.setattr(data, "DATA", tmp_path / "data")
    monkeypatch.setattr(data, "META", tmp_path / "data" / "metadata")
    candidate = {"id": "wc_12", "class": "apple", "source_id": "12", "source_split": "commons", "provider": "commons", "download_url": "https://example.test/photo.jpg"}
    calls = []

    def fetch(url, path):
        calls.append(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (500, 400), (90, 10, 50)).save(path)

    monkeypatch.setattr(data, "fetch_file", fetch)
    result = acquire.download_one(candidate)
    assert result["status"] == "pending_review" and result["crop_width"] == 500
    write_csv(data.META / "downloads.csv", [result], acquire.DOWNLOAD_FIELDS)
    monkeypatch.setattr(acquire, "candidates", lambda: [candidate])
    acquire.download(per_class=1, workers=1)
    assert len(calls) == 1
    assert read_csv(data.META / "downloads.csv")[0]["sha256"] == result["sha256"]


def test_source_too_small_remains_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire, "ROOT", tmp_path)
    monkeypatch.setattr(data, "DATA", tmp_path / "data")
    path = data.DATA / "raw" / "commons" / "commons" / "1.jpg"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100)).save(path)
    row = {"id": "wc_1", "class": "apple", "source_id": "1", "source_split": "commons", "provider": "commons"}
    result = acquire.download_one(row)
    assert result["status"] == "failed" and "too small" in result["error"]


def test_oi_inventory_joins_human_labels_and_attribution(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "META", tmp_path)
    labels = [{"ImageID": "123", "LabelName": "/m/0fp6w", "Confidence": "1", "Source": "verification"}]
    info = [{"ImageID": "123", "License": "https://creativecommons.org/licenses/by/2.0/", "Author": "Name", "OriginalLandingURL": "https://example.test/page", "OriginalURL": "https://example.test/original.jpg", "Thumbnail300KURL": "https://example.test/thumb.jpg"}]
    monkeypatch.setattr(data, "remote_csv", lambda url: iter(labels if url == acquire.LABEL_URLS["validation"] else info))
    acquire.oi_inventory(["validation"])
    rows = read_csv(tmp_path / "expanded_oi_validation.csv")
    assert rows[0]["id"] == "oi_123" and rows[0]["download_url"].endswith("thumb.jpg")
    assert rows[0]["eligible"] == "1"


def test_candidates_prefer_human_to_machine(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "META", tmp_path)
    for source in ("human_positive", "machine_NOT_ground_truth"):
        write_csv(tmp_path / f"expanded_{source}.csv", [{"id": "oi_123", "class": "apple", "eligible": "1", "label_source": source}], acquire.FIELDS)
    assert acquire.candidates()[0]["label_source"] == "human_positive"


def test_bbox_candidates_reused_with_correct_crop_downloader(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "META", tmp_path)
    item = {"id": "oi_123", "class": "apple", "source_id": "123", "source_split": "train", "eligible": "1"}
    write_csv(tmp_path / "candidates_train.csv", [item], data.CANDIDATE_FIELDS)
    rows = acquire.candidates()
    assert rows[0]["label_source"] == "human_bbox"
    assert rows[0]["download_url"].endswith("/train/123.jpg")
    monkeypatch.setattr(data, "download_one", lambda row: dict(row, bbox_called=True))
    assert acquire.download_one(rows[0])["bbox_called"]


def test_machine_inventory_human_negative_veto(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(data, "META", tmp_path)
    (tmp_path / "machine_label_targets_train.json").write_text(json.dumps({"good": {"class": "apple", "confidence": .9}, "bad": {"class": "apple", "confidence": .8}}))
    labels = [{"ImageID": "bad", "LabelName": "/m/014j1m", "Confidence": "0", "Source": "verification"}]
    infos = [{"ImageID": identifier, "License": "https://creativecommons.org/licenses/by/2.0/", "Author": "Name", "OriginalLandingURL": "https://example.test/page", "OriginalURL": "https://example.test/photo.jpg"} for identifier in ("good", "bad")]
    monkeypatch.setattr(data, "remote_csv", lambda url: iter(labels if url == acquire.LABEL_URLS["train"] else infos))
    acquire.machine_inventory()
    rows = read_csv(tmp_path / "expanded_oi_machine_train.csv")
    assert len(rows) == 1 and rows[0]["source_id"] == "good"
    assert rows[0]["label_source"] == "machine_NOT_ground_truth"


def test_download_lock_and_batch_target_do_not_skip_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "META", tmp_path)
    choices = [{"id": f"wc_{i}", "class": "apple", "eligible": "1"} for i in range(20)]
    calls = []
    monkeypatch.setattr(acquire, "candidates", lambda: choices)

    def fetch(row):
        calls.append(row["id"])
        return dict(row, status="pending_review" if len(calls) == 9 else "failed")

    monkeypatch.setattr(acquire, "download_one", fetch)
    acquire.download(per_class=1, workers=1)
    assert len(calls) == 9  # previous fixed-size offset incorrectly skipped 9 rows
    assert not (tmp_path / "expanded_download.lock").exists()
    (tmp_path / "expanded_download.lock").write_text("123")
    with pytest.raises(ValueError, match="lock exists"):
        acquire.download(per_class=1, workers=1)


def test_machine_stream_matches_chunk_boundaries_and_last_line(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "META", tmp_path)
    text = (b"ImageID,Source,LabelName,Confidence\n"
            b"0000000000000001,machine,/m/014j1m,0.9\n"
            b"0000000000000002,machine,/m/014j1m,0.49\n"
            b"0000000000000003,machine,/m/0fp6w,0.8")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            for i in range(0, len(text), 13):
                yield text[i:i + 13]

    monkeypatch.setattr(acquire.requests, "get", lambda *a, **kw: Response())
    infos = [{"ImageID": f"{i:016x}", "License": "https://creativecommons.org/licenses/by/2.0/", "Author": "Name", "OriginalLandingURL": "https://example.test/page", "OriginalURL": "https://example.test/photo.jpg"} for i in (1, 2, 3)]
    monkeypatch.setattr(data, "remote_csv", lambda url: iter([] if url == acquire.LABEL_URLS["train"] else infos))
    acquire.machine_inventory()
    rows = read_csv(tmp_path / "expanded_oi_machine_train.csv")
    assert {r["class"] for r in rows} == {"apple", "pineapple"}
    assert len(rows) == 2


def test_watcher_downloads_new_inventory_and_has_time_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "META", tmp_path)
    ticks = iter([0, 0, 61])
    calls = []
    monkeypatch.setattr(acquire.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(acquire.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(acquire, "download", lambda per_class, workers: calls.append((per_class, workers)))
    acquire.watch(hours=1 / 60, per_class=3, workers=1)
    assert calls == [(3, 1)]


def test_commons_pagination_subcategories_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "META", tmp_path)
    calls = []
    page = {"pageid": 123, "title": "File:Fruit.jpg", "imageinfo": [{"width": 640, "height": 480, "mime": "image/jpeg", "url": "https://example.test/photo.jpg", "descriptionurl": "https://example.test/page", "extmetadata": {"LicenseUrl": {"value": "https://creativecommons.org/licenses/by/4.0/"}, "Artist": {"value": "Name"}}}]}

    def api(params):
        calls.append(dict(params))
        if "pageids" in params:
            return {"query": {"pages": {"123": page}}}
        if params.get("cmcontinue"):
            return {"query": {"categorymembers": [{"pageid": 123, "ns": 6, "title": "File:Fruit.jpg"}]}}
        if params["cmtitle"] == "Category:Pineapples":
            return {"continue": {"cmcontinue": "next"}, "query": {"categorymembers": [{"pageid": 456, "ns": 14, "title": "Category:Pineapples in gardens"}]}}
        return {"query": {"categorymembers": [{"pageid": 123, "ns": 6, "title": "File:Fruit.jpg"}]}}

    monkeypatch.setattr(acquire, "api", api)
    acquire.commons_inventory(depth=1, max_categories=3, fruits=["pineapple"])
    rows = read_csv(tmp_path / "expanded_commons_pineapple.csv")
    assert len(rows) == 1 and rows[0]["eligible"] == "1"
    assert any(p.get("cmcontinue") == "next" for p in calls)
    assert any(p.get("cmtitle") == "Category:Pineapples in gardens" for p in calls)
    count = len(calls)
    acquire.commons_inventory(fruits=["pineapple"])
    assert len(calls) == count