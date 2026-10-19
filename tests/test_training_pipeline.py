from dataclasses import asdict
import hashlib
import json

import numpy as np
import pytest
from PIL import Image
import torch

from src import input_dataset, training_pipeline as pipeline
from src.common import CLASSES, read_csv, write_csv, write_json
from src.training_runtime import TrainConfig


def paired_fixture(root):
    rows = []
    rng = np.random.default_rng(123)
    for fruit in CLASSES:
        for index in range(12):
            split = "train" if index < 10 else "test"
            row = {"id": f"{fruit}_{index}", "class": fruit, "split": split,
                   "group_id": f"legacy_{split}_{fruit}"}
            pixels = Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
            for size, extension in ((224, "avif"), (32, "webp")):
                path = root / f"DataNew/Mid{size}/{split}/{fruit}/{row['id']}.{extension}"
                path.parent.mkdir(parents=True, exist_ok=True)
                pixels.resize((size, size)).save(path, quality=30)
                row[f"mid{size}_path"] = path.relative_to(root).as_posix()
                row[f"mid{size}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(row)
    write_csv(root / "data/metadata/splits.csv", rows)
    for size in (224, 32):
        write_json(root / f"configs/mid{size}.json", asdict(TrainConfig(size=size, epochs=2, batch_size=5, workers=0, device="cpu")))
    return rows


def test_one_entrypoint_trains_both_sizes_refits_and_tests_once(tmp_path, monkeypatch):
    paired_fixture(tmp_path)
    monkeypatch.setattr(input_dataset, "SPLIT_QUOTAS", {"train": 10, "test": 2})
    original_job = pipeline.train_job
    original_getitem = pipeline.ManifestDataset.__getitem__
    test_reads = []

    def guarded_job(*args, **kwargs):
        assert not test_reads, "Test must wait until all training plans finish"
        return original_job(*args, **kwargs)

    def track_reads(self, index):
        if self.rows[index]["split"] == "test":
            test_reads.append(self.rows[index]["id"])
        return original_getitem(self, index)

    monkeypatch.setattr(pipeline, "train_job", guarded_job)
    monkeypatch.setattr(pipeline.ManifestDataset, "__getitem__", track_reads)
    result = pipeline.run_pipeline(root=tmp_path, run_name="integration", validation_fraction=.2)
    output = tmp_path / "outputs/integration"
    assert set(result["results"]) == {"224", "32"}
    assert len(test_reads) == 20
    for size in (224, 32):
        dev = output / f"Mid{size}/development"
        final = output / f"Mid{size}/final"
        assert json.loads((dev / "settings.json").read_text())["train_images"] == 40
        assert json.loads((final / "settings.json").read_text())["train_images"] == 50
        assert {p.name for p in final.glob("*.pt")} == {"last.pt", "best.pt"}
        measured = json.loads((final / "test/metrics.json").read_text())
        assert measured["metrics"]["total"] == 10
        assert len(read_csv(final / "test/predictions.csv")) == 10
        assert (final / "test/confusion_matrix.png").exists()
        assert (dev / "learning_curves.png").exists()
    # Completed reruns must not evaluate held-out test a second time.
    monkeypatch.setattr(pipeline, "train_job", original_job)
    before = (output / "Mid224/final/test/metrics.json").read_bytes()
    pipeline.run_pipeline(root=tmp_path, run_name="integration", validation_fraction=.2)
    assert len(test_reads) == 20
    assert (output / "Mid224/final/test/metrics.json").read_bytes() == before


def test_check_only_audits_and_loads_without_creating_checkpoints(tmp_path, monkeypatch):
    paired_fixture(tmp_path)
    monkeypatch.setattr(input_dataset, "SPLIT_QUOTAS", {"train": 10, "test": 2})
    result = pipeline.run_pipeline(root=tmp_path, run_name="check", check_only=True)
    assert result["status"] == "preflight_passed"
    assert not list((tmp_path / "outputs").rglob("*.pt"))


def test_output_name_cannot_escape_project(tmp_path):
    with pytest.raises(ValueError, match="run name"):
        pipeline.run_pipeline(root=tmp_path, run_name="../outside")
