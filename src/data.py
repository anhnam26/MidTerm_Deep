"""Open Images inventory, selective downloads and human-reviewed dataset builds.

All downloaded crops remain PENDING until reviewed. No synthetic quota filling.
"""

import argparse
import collections
import concurrent.futures
import csv
import hashlib
import json
import random
import tempfile
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageOps

from .common import CLASSES, CLASS_IDS, ROOT, read_csv, write_csv, write_json

DATA = ROOT / "data"
META = DATA / "metadata"
# Decimal megabytes: total image file bytes across train/val/test for each set.
DATASET_LIMIT_BYTES = {224: 25_000_000, 32: 20_000_000}
SPLIT_QUOTAS = {"train": 4500, "val": 500, "test": 50}
BOX_URLS = {
    "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
}
INFO_URLS = {
    "train": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
    "validation": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    "test": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
}
HEADERS = {"User-Agent": "FruitMidterm/1.0 (educational selective image collection)"}
CANDIDATE_FIELDS = [
    "id", "class", "source_split", "source_id", "group_id", "xmin", "xmax",
    "ymin", "ymax", "box_area", "source_url", "landing_url", "license", "author",
    "title", "rotation", "eligible", "reason",
]
DOWNLOAD_FIELDS = CANDIDATE_FIELDS + [
    "status", "error", "original_path", "crop_path", "width", "height",
    "crop_width", "crop_height", "original_sha256", "sha256", "dhash",
    "provider", "download_url", "label_source", "label_confidence",
]


def remote_csv(url):
    """Stream official CSV without keeping gigabytes of unrelated annotations."""
    with requests.get(url, headers=HEADERS, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        lines = (line.decode("utf-8-sig") + "\n" for line in response.iter_lines(chunk_size=1024 * 1024))
        yield from csv.DictReader(lines)


def inventory(splits, refresh=False):
    META.mkdir(parents=True, exist_ok=True)
    summary_path = META / "inventory_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    for split in splits:
        output = META / f"candidates_{split}.csv"
        if output.exists() and not refresh:
            rows = read_csv(output)
            print(f"Reusing {output}", flush=True)
        else:
            start = time.monotonic()
            boxes_cache = META / f"filtered_boxes_{split}.json"
            if boxes_cache.exists() and not refresh:
                selected = json.loads(boxes_cache.read_text(encoding="utf-8"))
            else:
                targets = {}
                print(f"Streaming bounding boxes: {split}", flush=True)
                for index, box in enumerate(remote_csv(BOX_URLS[split]), 1):
                    fruit = CLASS_IDS.get(box["LabelName"])
                    if fruit:
                        item = targets.setdefault(box["ImageID"], {"classes": set(), "boxes": []})
                        item["classes"].add(fruit)
                        area = (float(box["XMax"]) - float(box["XMin"])) * (float(box["YMax"]) - float(box["YMin"]))
                        good = not any(box.get(flag, "0") == "1" for flag in ("IsDepiction", "IsGroupOf", "IsInside", "IsTruncated"))
                        if good and area >= 0.04:
                            item["boxes"].append((area, fruit, box))
                    if index % 250000 == 0:
                        write_json(META / "inventory_progress.json", {
                            "split": split, "phase": "boxes", "rows": index,
                            "target_images": len(targets), "elapsed_seconds": time.monotonic() - start,
                        })
                selected = {}
                for image_id, item in targets.items():
                    if len(item["classes"]) != 1 or not item["boxes"]:
                        continue
                    area, fruit, box = max(item["boxes"], key=lambda entry: entry[0])
                    selected[image_id] = {
                        "id": f"oi_{image_id}", "class": fruit, "source_split": split,
                        "source_id": image_id, "group_id": f"oi_{image_id}",
                        "xmin": box["XMin"], "xmax": box["XMax"], "ymin": box["YMin"], "ymax": box["YMax"],
                        "box_area": round(area, 5), "eligible": "0", "reason": "metadata_missing",
                    }
                write_json(boxes_cache, selected)
            print(f"Matching provenance metadata for {len(selected)} candidates: {split}", flush=True)
            found = 0
            for index, info in enumerate(remote_csv(INFO_URLS[split]), 1):
                row = selected.get(info["ImageID"])
                if row is not None:
                    license_url = info.get("License", "")
                    row.update({
                        "source_url": info.get("OriginalURL", ""),
                        "landing_url": info.get("OriginalLandingURL", ""),
                        "license": license_url, "author": info.get("Author", ""),
                        "title": info.get("Title", ""), "rotation": info.get("Rotation", ""),
                    })
                    # Require explicit attribution metadata. License still needs human verification.
                    eligible = license_url.rstrip("/") in (
                        "https://creativecommons.org/licenses/by/2.0",
                        "http://creativecommons.org/licenses/by/2.0",
                    ) and bool(row["author"] and row["landing_url"])
                    row["eligible"] = "1" if eligible else "0"
                    row["reason"] = "pending_license_and_label_review" if eligible else "license_or_attribution_missing"
                    found += 1
                    if found == len(selected):
                        break
                if index % 250000 == 0:
                    write_json(META / "inventory_progress.json", {"split": split, "phase": "provenance", "rows": index, "matched": found})
            rows = sorted(selected.values(), key=lambda row: row["id"])
            write_csv(output, rows, CANDIDATE_FIELDS)
        counts = collections.Counter(row["class"] for row in rows if row["eligible"] == "1")
        summary[split] = {fruit: counts[fruit] for fruit in CLASSES}
        write_json(META / "inventory_summary.json", summary)
        print(split, summary[split], flush=True)
    return summary


def dhash(image):
    pixels = list(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return f"{value:016x}"


def crop_box(image, row, margin=0.10):
    x0, x1 = float(row["xmin"]) * image.width, float(row["xmax"]) * image.width
    y0, y1 = float(row["ymin"]) * image.height, float(row["ymax"]) * image.height
    if not 0 <= x0 < x1 <= image.width or not 0 <= y0 < y1 <= image.height:
        raise ValueError("Invalid bounding box")
    dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
    return image.crop((max(0, int(x0 - dx)), max(0, int(y0 - dy)), min(image.width, int(x1 + dx + 0.999)), min(image.height, int(y1 + dy + 0.999))))


def fetch_file(url, path, attempts=3, max_bytes=25_000_000):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(attempts):
        try:
            with requests.get(url, headers=HEADERS, stream=True, timeout=(15, 60)) as response:
                if response.status_code in (429, 503):
                    retry_after = response.headers.get("Retry-After", "60")
                    delay = int(retry_after) if retry_after.isdigit() else 60
                    time.sleep(min(max(delay, 30), 300))
                response.raise_for_status()
                size = 0
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(65536):
                        size += len(chunk)
                        if size > max_bytes:
                            raise ValueError("Download exceeds original-file safety limit")
                        handle.write(chunk)
            with Image.open(partial) as image:
                image.verify()
            partial.replace(path)
            return
        except (requests.RequestException, OSError, ValueError):
            partial.unlink(missing_ok=True)
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def download_one(row):
    result = dict(row)
    original = DATA / "raw" / "openimages" / row["source_split"] / f"{row['source_id']}.jpg"
    crop = DATA / "crops" / row["class"] / f"{row['id']}.png"
    try:
        if not original.exists():
            fetch_file(f"https://open-images-dataset.s3.amazonaws.com/{row['source_split']}/{row['source_id']}.jpg", original)
        # CVDF images are pre-rotated to match boxes: don't apply Rotation a second time.
        with Image.open(original) as opened:
            image = opened.convert("RGB")
            cropped = crop_box(image, row)
        result.update({"width": image.width, "height": image.height, "crop_width": cropped.width, "crop_height": cropped.height})
        if min(cropped.size) < 224:
            raise ValueError("Crop too small for native Mid224 (minimum side <224)")
        crop.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(crop, format="PNG")
        result.update({
            "status": "pending_review", "error": "", "original_path": original.relative_to(ROOT).as_posix(),
            "crop_path": crop.relative_to(ROOT).as_posix(),
            "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
            "sha256": hashlib.sha256(crop.read_bytes()).hexdigest(), "dhash": dhash(cropped),
        })
    except Exception as error:
        result.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
    return result


def download(splits, per_class, workers, seed):
    rows = []
    for split in splits:
        rows.extend(read_csv(META / f"candidates_{split}.csv"))
    path = META / "downloads.csv"
    existing = {row["id"]: row for row in read_csv(path)} if path.exists() else {}
    rng = random.Random(seed)
    jobs = []
    for fruit in CLASSES:
        choices = [row for row in rows if row["class"] == fruit and row["eligible"] == "1"]
        rng.shuffle(choices)
        choices = choices[:per_class]
        jobs.extend(row for row in choices if not (
            row["id"] in existing and existing[row["id"]]["status"] == "pending_review"
            and (ROOT / existing[row["id"]]["crop_path"]).exists()
        ))
    print(f"Downloading up to {len(jobs)} images with {workers} workers; all remain pending human review.", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(download_one, jobs), 1):
            existing[result["id"]] = result
            if index % 10 == 0 or index == len(jobs):
                write_csv(path, sorted(existing.values(), key=lambda row: row["id"]), DOWNLOAD_FIELDS)
                print(f"{index}/{len(jobs)} attempted", flush=True)
    report = {fruit: dict(collections.Counter(row["status"] for row in existing.values() if row["class"] == fruit)) for fruit in CLASSES}
    write_json(META / "download_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def pilot_report():
    rows = read_csv(META / "downloads.csv")
    usable = [row for row in rows if row["status"] == "pending_review"]
    groups = group_duplicates(usable)
    duplicates = [[row["id"] for row in group] for group in groups if len(group) > 1]
    unique = collections.Counter(group[0]["class"] for group in groups if len({row["class"] for row in group}) == 1)
    reviewed_path = META / "reviews.csv"
    reviews = {row["id"]: row["decision"] for row in read_csv(reviewed_path)} if reviewed_path.exists() else {}
    report = {}
    for fruit in CLASSES:
        entries = [row for row in rows if row["class"] == fruit]
        downloaded = [row for row in usable if row["class"] == fruit]
        report[fruit] = {"attempted": len(entries), "downloaded_native_224": len(downloaded),
                         "acceptance_not_automatic": True, "human_accepted": sum(reviews.get(row["id"]) == "accept" for row in downloaded),
                         "failed": dict(collections.Counter(row["error"] for row in entries if row["status"] == "failed")),
                         "candidate_independent_groups_before_review": unique[fruit],
                         "crop_dimensions": [[int(row["crop_width"]), int(row["crop_height"])] for row in downloaded]}
    write_json(META / "pilot_report.json", {"classes": report, "near_duplicate_candidates": duplicates,
                                            "raw_bytes": sum(path.stat().st_size for path in (DATA / "raw").rglob("*.jpg")),
                                            "official_dataset_ready": False})
    print(json.dumps(report, indent=2))


def contact_sheets():
    rows = read_csv(META / "downloads.csv")
    for fruit in CLASSES:
        selected = [row for row in rows if row["class"] == fruit and row["status"] == "pending_review"]
        for offset in range(0, len(selected), 50):
            page = selected[offset:offset + 50]
            canvas = Image.new("RGB", (5 * 224, 10 * 260), "white")
            draw = ImageDraw.Draw(canvas)
            for index, row in enumerate(page):
                x, y = (index % 5) * 224, (index // 5) * 260
                with Image.open(ROOT / row["crop_path"]) as image:
                    canvas.paste(ImageOps.pad(image.convert("RGB"), (224, 224), method=Image.Resampling.LANCZOS), (x, y))
                draw.text((x + 3, y + 225), row["id"], fill="black")
                draw.text((x + 3, y + 242), f"{row['crop_width']}x{row['crop_height']}", fill="black")
            target = META / "contact_sheets" / f"{fruit}_{offset // 50 + 1:03d}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(target, quality=92)
            lowres = Image.new("RGB", canvas.size, "white")
            lowdraw = ImageDraw.Draw(lowres)
            for index, row in enumerate(page):
                x, y = (index % 5) * 224, (index // 5) * 260
                with Image.open(ROOT / row["crop_path"]) as opened:
                    image = opened.convert("RGB")
                    square = ImageOps.pad(image, (max(image.size),) * 2, method=Image.Resampling.LANCZOS, color=(127, 127, 127))
                    pixels32 = square.resize((32, 32), Image.Resampling.LANCZOS)
                    lowres.paste(pixels32.resize((224, 224), Image.Resampling.NEAREST), (x, y))
                lowdraw.text((x + 3, y + 225), row["id"], fill="black")
                lowdraw.text((x + 3, y + 242), "32px preview (nearest zoom)", fill="black")
            lowres.save(target.with_name(target.stem + "_32px.png"))


def import_local(manifest_path):
    """Import independently collected licensed photos using the same schema.

    Each row describes one source image, not augmented versions. group_id must
    identify the physical fruit/session/sequence where known. Human review is
    still mandatory; this command does not verify copyright on the author's behalf.
    """
    incoming = read_csv(manifest_path)
    required = {"id", "class", "path", "group_id", "source_url", "landing_url", "license", "author", "title"}
    if not incoming or not required.issubset(incoming[0]):
        raise ValueError(f"Import CSV must have fields: {sorted(required)}")
    path = META / "downloads.csv"
    existing = {row["id"]: row for row in read_csv(path)} if path.exists() else {}
    ids = set()
    for row in incoming:
        if not row["id"].startswith("local_") or not row["id"].replace("_", "").replace("-", "").isalnum():
            raise ValueError("Local IDs must use local_ prefix with safe alphanumeric/dash/underscore characters")
        if row["id"] in existing or row["id"] in ids or row["class"] not in CLASSES:
            raise ValueError(f"Duplicate/invalid id or class: {row['id']}")
        if any(not row[field].strip() for field in ("group_id", "source_url", "landing_url", "license", "author")):
            raise ValueError("Missing source attribution/license/related-image group")
        source = Path(row["path"])
        if not source.is_absolute():
            source = Path(manifest_path).resolve().parent / source
        if source.stat().st_size > 25_000_000:
            raise ValueError("Source image safety size exceeded")
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        if min(image.size) < 224:
            raise ValueError(f"Source too small: {source}")
        crop = DATA / "crops" / row["class"] / f"{row['id']}.png"
        crop.parent.mkdir(parents=True, exist_ok=True)
        image.save(crop)
        result = {field: row.get(field, "") for field in DOWNLOAD_FIELDS}
        result.update({"source_id": row["id"], "source_split": "local", "xmin": "0", "xmax": "1", "ymin": "0", "ymax": "1",
                       "box_area": "1", "eligible": "1", "reason": "pending_license_and_label_review", "rotation": "0",
                       "status": "pending_review", "error": "", "original_path": str(source.resolve()),
                       "crop_path": crop.relative_to(ROOT).as_posix(), "width": image.width, "height": image.height,
                       "crop_width": image.width, "crop_height": image.height, "original_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                       "sha256": hashlib.sha256(crop.read_bytes()).hexdigest(), "dhash": dhash(image)})
        existing[row["id"]] = result
        ids.add(row["id"])
        # Save after each image to preserve completed imports on interruption.
        write_csv(path, sorted(existing.values(), key=lambda item: item["id"]), DOWNLOAD_FIELDS)
    print(f"Imported {len(ids)} local source photos, pending visual/license review")


def review(ids_path, decision, note):
    ids = {line.strip() for line in Path(ids_path).read_text(encoding="utf-8").splitlines() if line.strip()}
    downloaded = {row["id"]: row for row in read_csv(META / "downloads.csv")}
    if not ids or any(item not in downloaded or downloaded[item]["status"] != "pending_review" for item in ids):
        raise ValueError("IDs must identify successfully downloaded pending-review crops")
    path = META / "reviews.csv"
    existing = {row["id"]: row for row in read_csv(path)} if path.exists() else {}
    for item in ids:
        existing[item] = {"id": item, "decision": decision, "note": note}
    write_csv(path, sorted(existing.values(), key=lambda row: row["id"]), ["id", "decision", "note"])


def group_duplicates(rows, distance=6):
    """Conservative dHash grouping; inspect matches, not proof of semantic equality.

    Seven bands guarantee a candidate pair for <=6 differing bits. Also group by
    parent ID and original checksum. Whole connected components stay together.
    """
    if not 0 <= distance <= 6:
        raise ValueError("Seven-band hash lookup supports Hamming thresholds from 0 to 6")
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(i, j):
        parent[find(i)] = find(j)

    exact = {}
    bands = collections.defaultdict(list)
    hashes = [int(row["dhash"], 16) for row in rows]
    for i, row in enumerate(rows):
        for field in ("group_id", "original_sha256", "sha256"):
            key = (field, row[field])
            if row[field] and key in exact:
                join(i, exact[key])
            exact[key] = i
        candidates = set()
        for band in range(7):
            key = (band, (hashes[i] >> (band * 9)) & 511)
            candidates.update(bands[key])
        for j in candidates:
            if (hashes[i] ^ hashes[j]).bit_count() <= distance:
                join(i, j)
        for band in range(7):
            bands[(band, (hashes[i] >> (band * 9)) & 511)].append(i)
    groups = collections.defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    return list(groups.values())


def development_reservations():
    combined = {field: set() for field in ("ids", "group_ids", "original_sha256", "sha256", "dhash")}
    for path in sorted(META.glob("compression_study_*/development_reservations.json")):
        reservation = json.loads(path.read_text(encoding="utf-8"))
        for field in combined:
            combined[field].update(reservation.get(field, []))
    return combined


def reserved_for_development(row, reservations):
    if any(row.get(field) and row[field] in reservations.get(key, ()) for field, key in
           (("id", "ids"), ("group_id", "group_ids"), ("original_sha256", "original_sha256"), ("sha256", "sha256"))):
        return True
    return bool(row.get("dhash")) and any((int(row["dhash"], 16) ^ int(value, 16)).bit_count() <= 6 for value in reservations.get("dhash", ()))


def select_splits(groups, train_per_class=4500, val_per_class=500, test_per_class=50, seed=2026, reservations=None):
    # Select only one representative per near-duplicate group: conservative quota.
    representatives = []
    for group in groups:
        if len({row["class"] for row in group}) > 1:
            continue  # conflicting labels require additional human investigation
        representative = max(group, key=lambda row: int(row["crop_width"]) * int(row["crop_height"]))
        result_row = dict(representative)
        if reservations and any(reserved_for_development(row, reservations) for row in group):
            result_row["development_only"] = "1"
        representatives.append(result_row)
    rng = random.Random(seed)
    result = []
    needed = train_per_class + val_per_class + test_per_class
    counts = collections.Counter(row["class"] for row in representatives)
    shortage = {fruit: needed - counts[fruit] for fruit in CLASSES if counts[fruit] < needed}
    if shortage:
        raise ValueError(f"Not enough reviewed independent groups. Missing: {shortage}. No quota filling allowed.")
    for fruit in CLASSES:
        choices = sorted((row for row in representatives if row["class"] == fruit), key=lambda row: row["id"])
        rng.shuffle(choices)
        # This is a new, custom classification benchmark, not Open Images evaluation.
        # Randomize independent groups across ALL sources; retain source_split only
        # as provenance. Never require 550 samples from tiny upstream held-out sets.
        eligible_test = [row for row in choices if row.get("development_only") != "1"]
        if len(eligible_test) < test_per_class:
            raise ValueError(f"Not enough unseen test groups for {fruit}; development reservations cannot enter test")
        testing = eligible_test[:test_per_class]
        test_ids = {row["id"] for row in testing}
        remaining = [row for row in choices if row["id"] not in test_ids]
        validation = remaining[:val_per_class]
        val_ids = {row["id"] for row in validation}
        training = remaining[val_per_class:val_per_class + train_per_class]
        picked = testing + validation
        for row in training + picked:
            row["split"] = "test" if row["id"] in test_ids else "val" if row["id"] in val_ids else "train"
            result.append(row)
    return result


def build(seed):
    if (META / "splits.csv").exists():
        raise ValueError("splits.csv already frozen; do not overwrite a held-out test set")
    if any((DATA / f"Mid{size}").exists() for size in DATASET_LIMIT_BYTES):
        raise ValueError("Existing Mid224/Mid32 directories; inspect them before building. No overwrite allowed.")
    reviews_path = META / "reviews.csv"
    if not reviews_path.exists():
        raise ValueError("No human reviews. Inspect crops AND source attribution, then record review decisions.")
    accepted = {row["id"] for row in read_csv(reviews_path) if row["decision"] == "accept"}
    rows = [row for row in read_csv(META / "downloads.csv") if row["id"] in accepted and row["status"] == "pending_review"]
    groups = group_duplicates(rows)
    write_json(META / "duplicate_groups.json", [[row["id"] for row in group] for group in groups if len(group) > 1])
    selected = select_splits(groups, seed=seed, reservations=development_reservations())
    DATA.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="paired_build_", dir=DATA) as temporary:
        staging = Path(temporary)
        totals = collections.Counter()
        for row in selected:
            crop_path = ROOT / row["crop_path"]
            if hashlib.sha256(crop_path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Crop changed after metadata/review: {row['id']}")
            with Image.open(crop_path) as opened:
                square = ImageOps.pad(opened.convert("RGB"), (max(opened.size),) * 2, method=Image.Resampling.LANCZOS, color=(127, 127, 127))
                for size in DATASET_LIMIT_BYTES:
                    path = staging / f"Mid{size}" / row["split"] / row["class"] / f"{row['id']}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    square.resize((size, size), Image.Resampling.LANCZOS).save(path)
                    totals[size] += path.stat().st_size
                    capacity_report(size, totals[size])  # Fail early; never freeze an oversized pair.
                    row[f"mid{size}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        report = validate_dataset(selected, staging)
        for size in DATASET_LIMIT_BYTES:
            (staging / f"Mid{size}").rename(DATA / f"Mid{size}")
    write_json(META / "build_settings.json", {"seed": seed, "classes": CLASSES, "resize": "RGB; preserve aspect; neutral pad; Lanczos; PNG", "quota": SPLIT_QUOTAS, "capacity_scope": "sum of image file bytes across train/val/test, not ZIP or allocated disk space", "capacity_unit": "MB (1,000,000 bytes)", "dataset_limit_bytes": DATASET_LIMIT_BYTES})
    write_json(META / "audit.json", report)
    # The frozen manifest is published only after both datasets pass validation.
    write_csv(META / "splits.csv", selected, DOWNLOAD_FIELDS + ["split", "development_only", "mid224_sha256", "mid32_sha256"])
    print(json.dumps(report, indent=2))


def capacity_report(size, total_bytes):
    limit = DATASET_LIMIT_BYTES[size]
    if total_bytes > limit:
        raise ValueError(f"Total dataset capacity limit exceeded for Mid{size}: {total_bytes} bytes > {limit} bytes ({limit / 1_000_000:g} MB)")
    return {"total_bytes": total_bytes, "total_mb": total_bytes / 1_000_000,
            "total_mib": total_bytes / 1024 ** 2,
            "limit_bytes": limit, "limit_mb": limit / 1_000_000,
            "limit_mib": limit / 1024 ** 2,
            "remaining_bytes": limit - total_bytes}


def validate_dataset(rows, dataset_root):
    reservations = development_reservations()
    if any(row.get("split") == "test" and (row.get("development_only") == "1" or reserved_for_development(row, reservations)) for row in rows):
        raise ValueError("Development/codec-study images cannot enter held-out test")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate IDs in splits")
    if len({row["original_sha256"] for row in rows}) != len(rows):
        raise ValueError("Duplicate parent content in splits")
    groups = group_duplicates(rows)
    if any(len(group) > 1 for group in groups):
        raise ValueError("Near duplicates or related parents present")
    report = {}
    for size in DATASET_LIMIT_BYTES:
        total_bytes = 0
        expected = set()
        counts = collections.Counter()
        for row in rows:
            if row["split"] not in SPLIT_QUOTAS or row["class"] not in CLASSES:
                raise ValueError(f"Invalid split/class: {row['id']}")
            path = dataset_root / f"Mid{size}" / row["split"] / row["class"] / f"{row['id']}.png"
            expected.add(path.resolve())
            with Image.open(path) as image:
                if image.size != (size, size) or image.mode != "RGB":
                    raise ValueError(f"Invalid mode/size: {path}")
                image.verify()
            if hashlib.sha256(path.read_bytes()).hexdigest() != row.get(f"mid{size}_sha256"):
                raise ValueError(f"Image content changed after dataset freeze: {path}")
            total_bytes += path.stat().st_size
            counts[row["split"], row["class"]] += 1
        actual = {path.resolve() for path in (dataset_root / f"Mid{size}").rglob("*") if path.is_file()}
        if actual != expected:
            raise ValueError(f"Unmanifested/missing images in Mid{size}")
        capacity = capacity_report(size, total_bytes)
        for split, quota in SPLIT_QUOTAS.items():
            if any(counts[split, fruit] != quota for fruit in CLASSES):
                raise ValueError(f"Wrong quota in Mid{size}/{split}")
        report[f"Mid{size}"] = {"images": len(rows), **capacity, "counts": {split: {fruit: counts[split, fruit] for fruit in CLASSES} for split in SPLIT_QUOTAS}}
    return report


def audit():
    report = validate_dataset(read_csv(META / "splits.csv"), DATA)
    write_json(META / "audit.json", report)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--splits", nargs="+", choices=list(BOX_URLS), default=["validation", "test", "train"])
    inv.add_argument("--refresh", action="store_true")
    down = sub.add_parser("download")
    down.add_argument("--splits", nargs="+", choices=list(BOX_URLS), default=["validation", "test"])
    down.add_argument("--per-class", type=int, default=25)
    down.add_argument("--workers", type=int, default=4)
    down.add_argument("--seed", type=int, default=2026)
    sub.add_parser("sheets")
    sub.add_parser("pilot-report")
    imp = sub.add_parser("import-local")
    imp.add_argument("--manifest", type=Path, required=True)
    rev = sub.add_parser("review")
    rev.add_argument("--ids", type=Path, required=True)
    rev.add_argument("--decision", choices=["accept", "reject"], required=True)
    rev.add_argument("--note", required=True, help="Confirm visual label AND license verification, or state rejection reason")
    bld = sub.add_parser("build")
    bld.add_argument("--seed", type=int, default=2026)
    sub.add_parser("audit")
    args = parser.parse_args()
    if args.command == "inventory":
        inventory(args.splits, args.refresh)
    elif args.command == "download":
        if args.per_class < 1 or not 1 <= args.workers <= 8:
            parser.error("per-class must be positive; workers must be 1..8")
        download(args.splits, args.per_class, args.workers, args.seed)
    elif args.command == "sheets":
        contact_sheets()
    elif args.command == "pilot-report":
        pilot_report()
    elif args.command == "import-local":
        import_local(args.manifest)
    elif args.command == "review":
        review(args.ids, args.decision, args.note)
    elif args.command == "build":
        build(args.seed)
    elif args.command == "audit":
        audit()


if __name__ == "__main__":
    main()