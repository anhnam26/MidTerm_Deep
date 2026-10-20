"""Validate the paired AVIF/WebP input bundle without weakening legacy PNG gates."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from PIL import AvifImagePlugin, Image

from .common import CLASSES, ROOT, read_csv, write_json

AvifImagePlugin.MAX_THREADS = 1
SPLIT_QUOTAS = {"train": 5000, "test": 50}
LIMITS = {224: 25_000_000, 32: 20_000_000}
FORMATS = {224: ("AVIF", ".avif"), 32: ("WEBP", ".webp")}


def image_path(root, row, size):
    root = Path(root).resolve()
    supplied = Path(row[f"mid{size}_path"])
    scope = root / "DataNew" / f"Mid{size}"
    path = (root / supplied).resolve()
    if (
        supplied.is_absolute()
        or not path.is_relative_to(scope)
        or path.suffix.lower() != FORMATS[size][1]
        or path.parent.name != row["class"]
    ):
        raise ValueError(f"Unsafe image path: {supplied}")
    return path


def validate(root, rows):
    root = Path(root)
    identifiers, groups = set(), {}
    counts = Counter()
    for row in rows:
        if row["id"] in identifiers:
            raise ValueError(f"Duplicate identity: {row['id']}")
        identifiers.add(row["id"])
        if row["class"] not in CLASSES or row["split"] not in SPLIT_QUOTAS:
            raise ValueError(f"Invalid class/split: {row['id']}")
        group = row.get("group_id")
        if not group:
            raise ValueError(f"Missing capture group: {row['id']}")
        if group in groups and groups[group] != row["split"]:
            raise ValueError(f"Capture group crosses splits: {group}")
        groups[group] = row["split"]
        counts[row["split"], row["class"]] += 1
    for split, quota in SPLIT_QUOTAS.items():
        if any(counts[split, fruit] != quota for fruit in CLASSES):
            raise ValueError(f"Wrong quota for {split}; expected {quota} per class")
    report = {
        "layout": "paired-compressed-v1",
        "structural_pass": True,
        "identities": len(rows),
        "capture_groups": len(groups),
        "counts": {s: {c: counts[s, c] for c in CLASSES} for s in SPLIT_QUOTAS},
        "review_scope": "Technical validation; source and visual-review evidence is recorded separately.",
    }
    for size, limit in LIMITS.items():
        expected, file_hashes, pixel_hashes = set(), set(), set()
        total_bytes = 0
        last_progress = time.monotonic()
        for index, row in enumerate(rows, 1):
            path = image_path(root, row, size)
            if not path.is_file():
                raise ValueError(f"Missing paired image: {path}")
            if path in expected:
                raise ValueError(f"Duplicate output path: {path}")
            expected.add(path)
            with Image.open(path) as image:
                image.load()
                if (
                    image.size != (size, size)
                    or image.mode != "RGB"
                    or getattr(image, "n_frames", 1) != 1
                    or image.format != FORMATS[size][0]
                ):
                    raise ValueError(f"Wrong image geometry/mode/format: {path}")
                pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row.get(f"mid{size}_sha256"):
                raise ValueError(f"Image hash changed after freeze: {path}")
            if digest in file_hashes or pixel_hash in pixel_hashes:
                raise ValueError(f"Duplicate encoded or decoded image: {path}")
            file_hashes.add(digest)
            pixel_hashes.add(pixel_hash)
            total_bytes += path.stat().st_size
            if index % 5000 == 0 or time.monotonic() - last_progress >= 10:
                print(f"INPUT_AUDIT Mid{size}: {index}/{len(rows)}", flush=True)
                last_progress = time.monotonic()
        actual = {p.resolve() for p in (root / "DataNew" / f"Mid{size}").rglob("*") if p.is_file()}
        if actual != expected:
            raise ValueError(f"Unmanifested or missing files in Mid{size}")
        if total_bytes > limit:
            raise ValueError(f"Mid{size} capacity exceeded: {total_bytes} > {limit}")
        report[f"Mid{size}"] = {
            "images": len(rows),
            "total_bytes": total_bytes,
            "total_mb": total_bytes / 1_000_000,
            "limit_bytes": limit,
            "capacity_pass": True,
        }
    return report


def audit(root=ROOT):
    root = Path(root)
    report = validate(root, read_csv(root / "data/metadata/splits.csv"))
    write_json(root / "results/input_validation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit"])
    parser.parse_args()
    print(json.dumps(audit(), indent=2))


if __name__ == "__main__":
    main()
