import hashlib

import pytest
from PIL import Image

from src import compression, data
from src.common import read_csv, write_csv


def candidate(identifier, fruit="apple", provider="openimages"):
    return {"id": identifier, "class": fruit, "status": "pending_review", "provider": provider,
            "label_source": "human_bbox", "group_id": identifier, "original_sha256": identifier}


def test_sampling_seed_order_and_reviews():
    rows = [candidate(str(i)) for i in range(20)] + [candidate("failed"), candidate("reject")]
    rows[-2]["status"] = "failed"
    sample = compression.sample_candidates(rows, {"reject": "reject"}, 5)
    assert sample == compression.sample_candidates(list(reversed(rows)), {"reject": "reject"}, 5)
    assert len(sample) == 5
    assert not {"failed", "reject"} & {r["id"] for r in sample}
    assert sample != compression.sample_candidates(rows, {"reject": "reject"}, 5, seed=7)


def test_sampling_class_and_source_allocations():
    rows = [candidate(str(i)) for i in range(8)] + [candidate(f"wc{i}", provider="commons") for i in range(2)]
    rows += [candidate("banana", "banana")]
    sample = compression.sample_candidates(rows, {}, 5)
    assert len(sample) == 6
    assert sum(r["provider"] == "commons" for r in sample) == 1
    assert sum(r["class"] == "banana" for r in sample) == 1


def test_sampling_invalid_input():
    with pytest.raises(ValueError, match="Duplicate"):
        compression.sample_candidates([candidate("a"), candidate("a")], {})
    with pytest.raises(ValueError, match="positive"):
        compression.sample_candidates([], {}, 0)


def test_codec_decode_dimensions_rgb():
    image = Image.new("RGB", (224, 224), "orange")
    for name, codec, settings in compression.codec_configs():
        payload, decoded, rmse = compression.encode_image(image, codec, settings)
        assert len(payload) > 0 and decoded.mode == "RGB" and decoded.size == (224, 224)
        assert rmse >= 0
        if name == "png":
            assert rmse == 0


def test_projection_is_total_not_per_image(monkeypatch):
    monkeypatch.setattr(compression, "CLASSES", ("apple",))
    monkeypatch.setattr(data, "SPLIT_QUOTAS", {"train": 10, "test": 1})
    monkeypatch.setattr(data, "DATASET_LIMIT_BYTES", {224: 1000})
    result = compression.summarize([{"bytes": 100, "class": "apple", "rgb_rmse": 0}], 224)
    assert result["projected_bytes"] == 1100
    assert not result["projected_within_limit"]


@pytest.mark.parametrize("tamper", [False, True])
def test_study_read_only_and_unapproved(tmp_path, monkeypatch, tamper):
    monkeypatch.setattr(compression, "ROOT", tmp_path)
    monkeypatch.setattr(data, "META", tmp_path / "metadata")
    crop = tmp_path / "crop.png"
    Image.new("RGB", (300, 240), "red").save(crop)
    row = candidate("a")
    row.update({"crop_path": "crop.png", "sha256": "bad" if tamper else hashlib.sha256(crop.read_bytes()).hexdigest()})
    write_csv(data.META / "downloads.csv", [row])
    write_csv(data.META / "reviews.csv", [{"id": "other", "decision": "reject"}])
    before = {p: p.read_bytes() for p in (crop, data.META / "downloads.csv", data.META / "reviews.csv")}
    output = tmp_path / "study"
    report = compression.study(output, per_class=1, preview_per_class=1)
    assert not report["auto_accept"] and not report["quality_verified"] and not report["official_dataset_ready"]
    assert bool(report["errors"]) == tamper
    assert all(p.read_bytes() == content for p, content in before.items())
    assert read_csv(output / "review_template.csv")[0]["label_ok"] == ""
    assert (output / "development_reservations.json").exists()
    assert not (data.META / "splits.csv").exists()
    if not tamper:
        with Image.open(output / "comparison_apple.png") as image:
            assert image.size == (6 * 224, 270)
    with pytest.raises(ValueError, match="No overwrite"):
        compression.study(output, per_class=1, preview_per_class=1)


def test_no_candidates_no_output(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "META", tmp_path / "metadata")
    write_csv(data.META / "downloads.csv", [], ["id", "class", "status"])
    output = tmp_path / "study"
    with pytest.raises(ValueError, match="No downloaded"):
        compression.study(output)
    assert not output.exists()