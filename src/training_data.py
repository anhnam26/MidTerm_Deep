"""A frozen development partition of training images; never repurpose test data."""

from collections import Counter, defaultdict
import hashlib
import random

from .common import CLASSES


def capture_key(row):
    group = row.get("group_id", "")
    verified = str(row.get("physical_groups_verified", "")).lower() in ("true", "1")
    # These identifiers describe an entire import, not one photographed specimen.
    aggregate = group.startswith("legacy_") or group in (
        "bananareal_author_train", "strawshape_training_corpus")
    if group and not aggregate and (verified or group.startswith("pqc500_specimen_")):
        return f"capture:{group}", True
    identity = row.get("source_identity") or row["id"]
    return f"unverified:{row.get('source_dataset', '')}:{row['class']}:{identity}", False


def development_split(rows, fraction=.1, seed=2026):
    if not 0 < fraction < .5:
        raise ValueError("Validation fraction must be between 0 and 0.5")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate training identities")
    if any(r["class"] not in CLASSES or r["split"] not in ("train", "test") for r in rows):
        raise ValueError("Expected five classes and original train/test manifest")
    groups = defaultdict(list)
    test_keys = {capture_key(r)[0] for r in rows if r["split"] == "test"}
    for row in rows:
        if row["split"] == "train":
            key, _ = capture_key(row)
            if key in test_keys:
                raise ValueError(f"Source/capture group intersects reserved test: {key}")
            groups[key].append(row)
    if any(len({r["class"] for r in group}) != 1 for group in groups.values()):
        raise ValueError("Capture group has conflicting classes")
    val_ids = set()
    for fruit in CLASSES:
        candidates = sorted(k for k, group in groups.items() if group[0]["class"] == fruit)
        if len(candidates) < 2:
            raise ValueError(f"Not enough independent split groups for {fruit}")
        random.Random(f"{seed}:{fruit}").shuffle(candidates)
        total = sum(len(groups[key]) for key in candidates)
        target = max(1, round(total * fraction))
        selected, count = [], 0
        for key in candidates:
            length = len(groups[key])
            if count + length < total and abs(count + length - target) < abs(count - target):
                selected.append(key)
                count += length
        if not selected:
            raise ValueError(f"Cannot reserve validation groups for {fruit}")
        val_ids.update(r["id"] for key in selected for r in groups[key])
    train = sorted((r for r in rows if r["split"] == "train" and r["id"] not in val_ids), key=lambda r: r["id"])
    val = sorted((r for r in rows if r["id"] in val_ids), key=lambda r: r["id"])
    assignments = [{"id": r["id"], "class": r["class"],
                    "partition": "test_reserved" if r["split"] == "test" else "val" if r["id"] in val_ids else "train",
                    "capture_key": capture_key(r)[0], "capture_verified": capture_key(r)[1]}
                   for r in sorted(rows, key=lambda r: r["id"])]
    digest = hashlib.sha256("\n".join(f"{r['id']}:{r['partition']}:{r['capture_key']}" for r in assignments).encode()).hexdigest()
    unverified = sum(not capture_key(r)[1] for r in train + val)
    return train, val, {"seed": seed, "fraction_requested": fraction, "partition_sha256": digest,
                        "train_counts": dict(Counter(r["class"] for r in train)),
                        "validation_counts": dict(Counter(r["class"] for r in val)),
                        "test_reserved": sum(r["split"] == "test" for r in rows),
                        "unverified_capture_images": unverified,
                        "physical_independence_verified": unverified == 0,
                        "limitation": "Known specimens stay together. Legacy/import-wide groups use source identity; unknown near duplicates can inflate validation.",
                        "assignments": assignments}
