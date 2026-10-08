import hashlib
import argparse
import json

import pytest
import torch
from PIL import Image

from src.common import CLASSES, read_csv, write_csv
from src.data import crop_box, dhash, group_duplicates, select_splits
from src import data
from src import train
from src.model import FruitLite
from src.profile import profile
from src.train import learning_rate


@pytest.mark.parametrize("size", [32, 224])
def test_model_limits_and_shape(size):
    result = profile(FruitLite(), size)
    assert result["eligible"]
    assert result["output_shape"] == [1, 5]
    assert result["flops"] < 800_000_000


@pytest.mark.parametrize("multiscale,attention", [(True, True), (False, True), (True, False)])
def test_backward_and_ablation(multiscale, attention):
    torch.set_num_threads(2)
    model = FruitLite(multiscale=multiscale, attention=attention)
    result = model(torch.randn(2, 3, 32, 32))
    torch.nn.functional.cross_entropy(result, torch.tensor([0, 4])).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_crop_and_hash():
    image = Image.new("RGB", (500, 400), (200, 100, 0))
    crop = crop_box(image, {"xmin": "0.2", "xmax": "0.8", "ymin": "0.1", "ymax": "0.9"}, margin=0)
    assert crop.size == (300, 320)
    assert len(dhash(crop)) == 16
    with pytest.raises(ValueError):
        crop_box(image, {"xmin": "0.8", "xmax": "0.2", "ymin": "0", "ymax": "1"})


def row(identifier, fruit="apple", split="train", hash_value="0000000000000000"):
    return {"id": identifier, "class": fruit, "group_id": identifier, "source_split": split,
            "original_sha256": hashlib.sha256(identifier.encode()).hexdigest(), "sha256": identifier,
            "dhash": hash_value, "crop_width": "300", "crop_height": "300"}


def test_near_duplicates_and_parent_groups():
    rows = [row("a"), row("b", hash_value="0000000000000001"), row("c", hash_value="ffffffffffffffff")]
    groups = group_duplicates(rows)
    assert sorted(map(len, groups)) == [1, 2]
    rows[2]["group_id"] = "a"
    assert len(group_duplicates(rows)) == 1


def test_split_refuses_shortage():
    with pytest.raises(ValueError, match="No quota filling"):
        select_splits([[row("a")]])


def test_group_split_reproducible_and_disjoint():
    groups = []
    for fruit in CLASSES:
        for index, split in enumerate(["train", "train", "validation", "test"]):
            groups.append([row(f"{fruit}_{index}", fruit, split)])
    a = select_splits(groups, train_per_class=2, val_per_class=1, test_per_class=1)
    b = select_splits(groups, train_per_class=2, val_per_class=1, test_per_class=1)
    assert a == b
    assert len({item["id"] for item in a}) == 20
    train_ids = {item["id"] for item in a if item["split"] == "train"}
    assert train_ids.isdisjoint(item["id"] for item in a if item["split"] in ("val", "test"))
    for fruit in CLASSES:
        assert sum(item["class"] == fruit and item["split"] == "test" for item in a) == 1


def test_conflicting_labels_not_counted():
    with pytest.raises(ValueError, match="Not enough"):
        select_splits([[row("a", "apple"), row("b", "banana")]], train_per_class=1, val_per_class=1, test_per_class=1)


def test_development_reservation_whole_group(monkeypatch):
    monkeypatch.setattr(data, "CLASSES", ("apple",))
    small = row("studied")
    large = row("representative")
    large["crop_width"] = "500"
    groups = [[small, large], [row("unseen")], [row("unseen2")]]
    splits = select_splits(groups, 1, 1, 1, reservations={"ids": {"studied"}})
    reserved = next(r for r in splits if r["id"] == "representative")
    assert reserved["development_only"] == "1" and reserved["split"] != "test"
    with pytest.raises(ValueError, match="unseen test groups"):
        select_splits(groups, 1, 1, 1, reservations={"ids": {"studied", "unseen", "unseen2"}})


@pytest.mark.parametrize("field,key", [("group_id", "group_ids"), ("original_sha256", "original_sha256"), ("sha256", "sha256"), ("dhash", "dhash")])
def test_development_reservation_related_image(field, key):
    item = row("a")
    value = "0000000000000001" if field == "dhash" else item[field]
    assert data.reserved_for_development(item, {key: {value}})


def test_reservation_loading_and_audit_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "META", tmp_path)
    folder = tmp_path / "compression_study_fixture"
    folder.mkdir()
    (folder / "development_reservations.json").write_text(json.dumps({"ids": ["a"]}))
    assert "a" in data.development_reservations()["ids"]
    item = row("a")
    item["split"] = "test"
    with pytest.raises(ValueError, match="codec-study"):
        data.validate_dataset([item], tmp_path)


def test_csv_attribution_roundtrip(tmp_path):
    rows = [{"author": 'Name, "quoted"', "title": "line one\nline two"}]
    path = tmp_path / "metadata.csv"
    write_csv(path, rows)
    assert read_csv(path) == rows


def test_lr_warmup_and_decay():
    assert learning_rate(0, 150, 5, .05) == pytest.approx(.01)
    assert learning_rate(4, 150, 5, .05) == pytest.approx(.05)
    assert learning_rate(149, 150, 5, .05) == pytest.approx(.0005)


def test_multiline_remote_metadata(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        def iter_lines(self, **kwargs):
            yield from [b'ImageID,Author,Title', b'abc,Writer,"line one', b'line two"']

    monkeypatch.setattr(data.requests, "get", lambda *args, **kwargs: Response())
    assert list(data.remote_csv("https://example.test/metadata.csv"))[0]["Title"] == "line one\nline two"


def test_local_import_and_review_are_not_automatic(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(data, "DATA", tmp_path / "data")
    monkeypatch.setattr(data, "META", tmp_path / "data" / "metadata")
    image = tmp_path / "source.png"
    Image.new("RGB", (300, 240), "red").save(image)
    manifest = tmp_path / "local.csv"
    write_csv(manifest, [{"id": "local_a", "class": "apple", "path": str(image), "group_id": "fruit-session-1",
                          "source_url": image.as_uri(), "landing_url": image.as_uri(), "license": "CC BY 4.0 (owner declaration)",
                          "author": "Test fixture", "title": "Synthetic test fixture, not actual fruit"}])
    data.import_local(manifest)
    rows = read_csv(data.META / "downloads.csv")
    assert rows[0]["status"] == "pending_review"
    assert rows[0]["group_id"] == "fruit-session-1"
    assert not (data.META / "reviews.csv").exists()
    ids = tmp_path / "ids.txt"
    ids.write_text("local_a\n", encoding="utf-8")
    data.review(ids, "reject", "Test fixture: not fruit")
    assert read_csv(data.META / "reviews.csv")[0]["decision"] == "reject"
    with pytest.raises(ValueError, match="Duplicate"):
        data.import_local(manifest)


def test_small_paired_build_and_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(data, "DATA", tmp_path / "data")
    monkeypatch.setattr(data, "META", tmp_path / "data" / "metadata")
    image = tmp_path / "crop.png"
    Image.new("RGB", (300, 240), "red").save(image)
    item = row("test_fixture")
    item.update({"crop_path": "crop.png", "sha256": hashlib.sha256(image.read_bytes()).hexdigest(), "status": "pending_review", "split": "train"})
    write_csv(data.META / "downloads.csv", [item])
    write_csv(data.META / "reviews.csv", [{"id": item["id"], "decision": "accept"}])
    # Reduce quota ONLY inside this test; production build remains strict.
    monkeypatch.setattr(data, "select_splits", lambda *args, **kwargs: [item])
    monkeypatch.setattr(data, "CLASSES", ("apple",))
    monkeypatch.setattr(data, "SPLIT_QUOTAS", {"train": 1, "val": 0, "test": 0})
    data.build(2026)
    for size in (32, 224):
        path = data.DATA / f"Mid{size}" / "train" / "apple" / "test_fixture.png"
        with Image.open(path) as opened:
            assert opened.mode == "RGB" and opened.size == (size, size)
    assert read_csv(data.META / "splits.csv")[0]["id"] == "test_fixture"
    settings = json.loads((data.META / "build_settings.json").read_text())
    assert settings["dataset_limit_bytes"] == {"224": 25_000_000, "32": 20_000_000}
    assert settings["capacity_unit"] == "MB (1,000,000 bytes)"
    report = data.audit()
    assert report["Mid224"]["limit_bytes"] == 25_000_000
    assert report["Mid32"]["limit_bytes"] == 20_000_000
    assert report["Mid224"]["limit_mb"] == 25
    assert report["Mid32"]["limit_mb"] == 20
    with pytest.raises(ValueError, match="already frozen"):
        data.build(2026)


def test_audit_refuses_wrong_size(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(data, "DATA", tmp_path / "data")
    monkeypatch.setattr(data, "META", tmp_path / "data" / "metadata")
    item = row("a")
    item["split"] = "train"
    write_csv(data.META / "splits.csv", [item])
    image = data.DATA / "Mid224" / "train" / "apple" / "a.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(image)
    with pytest.raises(ValueError, match="Invalid mode/size"):
        data.audit()


def test_evaluation_exports_metrics_and_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(data, "audit", lambda: None)
    rows = []
    for index, fruit in enumerate(CLASSES):
        item = {"id": f"fixture_{index}", "class": fruit, "split": "val"}
        rows.append(item)
        image = tmp_path / "data" / "Mid32" / "val" / fruit / f"fixture_{index}.png"
        image.parent.mkdir(parents=True)
        Image.new("RGB", (32, 32), (index * 40, 128, 64)).save(image)
    write_csv(tmp_path / "data" / "metadata" / "splits.csv", rows)
    model = FruitLite()
    checkpoint = tmp_path / "fixture.pt"
    settings = {"purpose": "development", "split_sha256": train.split_digest()}
    torch.save({"model": model.state_dict(), "model_settings": {}, "size": 32, "epoch": 1, "settings": settings}, checkpoint)
    args = argparse.Namespace(checkpoint=checkpoint, split="val", output=tmp_path / "evaluation", batch_size=5, workers=0)
    train.evaluate(args)
    result = json.loads((args.output / "metrics.json").read_text())
    assert result["metrics"]["total"] == 5
    assert len(result["confusion_matrix"]) == 5
    assert len(read_csv(args.output / "predictions.csv")) == 5
    args.output = tmp_path / "forbidden_test"
    args.split = "test"
    with pytest.raises(ValueError, match="locked final"):
        train.evaluate(args)
    settings["purpose"] = "SMOKE_ONLY_NOT_MIDTERM_RESULTS"
    torch.save({"settings": settings}, checkpoint)
    with pytest.raises(ValueError, match="smoke checkpoint"):
        train.evaluate(args)


def test_inventory_filters_labels_boxes_and_preserves_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "META", tmp_path)
    (tmp_path / "inventory_summary.json").write_text('{"test": {"apple": 7}}')
    boxes = []
    for image_id, label, depiction, area_side in [
        ("good", "/m/014j1m", "0", "0.8"),
        ("mixed", "/m/014j1m", "0", "0.8"),
        ("mixed", "/m/09qck", "0", "0.8"),
        ("drawing", "/m/014j1m", "1", "0.8"),
        ("small", "/m/014j1m", "0", "0.1"),
        ("unlicensed", "/m/014j1m", "0", "0.8"),
    ]:
        boxes.append({"ImageID": image_id, "LabelName": label, "XMin": "0", "XMax": area_side,
                      "YMin": "0", "YMax": area_side, "IsDepiction": depiction})
    infos = [{"ImageID": image_id, "License": license_url, "Author": "Photographer", "OriginalLandingURL": "https://example.test/photo", "OriginalURL": "https://example.test/photo.jpg"}
             for image_id, license_url in [("good", "https://creativecommons.org/licenses/by/2.0/"), ("unlicensed", "All rights reserved")]]
    calls = []

    def remote(url):
        calls.append(url)
        return iter(boxes if url == data.BOX_URLS["validation"] else infos)

    monkeypatch.setattr(data, "remote_csv", remote)
    summary = data.inventory(["validation"])
    assert summary["validation"]["apple"] == 1
    assert summary["test"]["apple"] == 7
    candidates = read_csv(tmp_path / "candidates_validation.csv")
    assert {item["source_id"] for item in candidates} == {"good", "unlicensed"}
    data.inventory(["validation"])
    assert len(calls) == 2  # completed CSV is reused without network