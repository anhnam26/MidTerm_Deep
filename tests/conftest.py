"""Data fixtures and bounded CPU threads for the data-free test suite."""

import csv
import hashlib
import json

from PIL import Image
import pytest
import torch

from src import input_dataset as inputs


torch.set_num_threads(2)


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(inputs, "CLASSES", ("apple",))
    monkeypatch.setattr(inputs, "SPLIT_QUOTAS", {"train": 1, "test": 1})
    rows = []
    for split, color in [("train", (180, 20, 30)), ("test", (20, 100, 180))]:
        row = {"id": "apple_" + split, "class": "apple", "split": split, "group_id": "capture_" + split}
        for size, extension in [(224, ".avif"), (32, ".webp")]:
            path = tmp_path / "DataNew" / f"Mid{size}" / split / "apple" / (row["id"] + extension)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (size, size), color).save(path)
            row[f"mid{size}_path"] = path.relative_to(tmp_path).as_posix()
            row[f"mid{size}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(row)
    metadata = tmp_path / "data/metadata"
    metadata.mkdir(parents=True)
    (metadata / "build_settings.json").write_text(json.dumps({"layout": "paired-compressed-v1"}))
    with (metadata / "splits.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path, rows
