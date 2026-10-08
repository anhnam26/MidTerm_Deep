"""Supplemental licensed-source acquisition. Candidates are NEVER auto-accepted."""

import argparse
import collections
import concurrent.futures
import hashlib
import html
import json
import os
import random
import re
import shutil
import time

import requests
from PIL import Image, ImageOps

from . import data
from .common import CLASS_IDS, CLASSES, ROOT, read_csv, write_csv, write_json

LABEL_URLS = {split: f"https://storage.googleapis.com/openimages/v7/oidv7-{name}-annotations-human-imagelabels.csv"
              for split, name in (("train", "train"), ("validation", "val"), ("test", "test"))}
INFO_URLS = dict(data.INFO_URLS, train="https://storage.googleapis.com/openimages/v6/oidv6-train-images-with-labels-with-rotation.csv")
FIELDS = data.CANDIDATE_FIELDS + ["provider", "download_url", "label_source", "label_confidence"]
DOWNLOAD_FIELDS = data.DOWNLOAD_FIELDS
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
CATEGORIES = {"apple": "Apples", "banana": "Bananas", "orange": "Oranges", "strawberry": "Strawberries", "pineapple": "Pineapples"}


def positive_targets(labels):
    targets = collections.defaultdict(set)
    for row in labels:
        fruit = CLASS_IDS.get(row["LabelName"])
        if fruit and row.get("Confidence") == "1" and row.get("Source") in ("verification", "crowdsource-verification"):
            targets[row["ImageID"]].add(fruit)
    return {identifier: next(iter(fruits)) for identifier, fruits in targets.items() if len(fruits) == 1}


def oi_inventory(splits):
    for split in splits:
        output = data.META / f"expanded_oi_{split}.csv"
        if output.exists():
            print(f"Reusing {output}", flush=True)
            continue
        cache = data.META / f"image_label_targets_{split}.json"
        if cache.exists():
            targets = json.loads(cache.read_text(encoding="utf-8"))
        else:
            def labels():
                for index, row in enumerate(data.remote_csv(LABEL_URLS[split]), 1):
                    if index % 1000000 == 0:
                        write_json(data.META / "expanded_inventory_progress.json", {"split": split, "phase": "labels", "rows": index})
                        print(f"{split}: {index:,} label rows", flush=True)
                    yield row
            targets = positive_targets(labels())
            write_json(cache, targets)
        print(f"{split}: human-positive single-target candidates {dict(collections.Counter(targets.values()))}", flush=True)
        rows = []
        for index, info in enumerate(data.remote_csv(INFO_URLS[split]), 1):
            identifier = info["ImageID"]
            if identifier in targets:
                license_url = info.get("License", "")
                eligible = license_url.rstrip("/") in ("https://creativecommons.org/licenses/by/2.0", "http://creativecommons.org/licenses/by/2.0")
                eligible = eligible and bool(info.get("Author") and info.get("OriginalLandingURL"))
                rows.append({"id": f"oi_{identifier}", "class": targets[identifier], "source_split": split,
                             "source_id": identifier, "group_id": f"oi_{identifier}", "xmin": "0", "xmax": "1", "ymin": "0", "ymax": "1", "box_area": "1",
                             "source_url": info.get("OriginalURL", ""), "landing_url": info.get("OriginalLandingURL", ""),
                             "license": license_url, "author": info.get("Author", ""), "title": info.get("Title", ""),
                             "rotation": info.get("Rotation", ""), "eligible": "1" if eligible else "0",
                             "reason": "whole_image_human_label_pending_visual_license_review", "provider": "openimages",
                             "download_url": info.get("Thumbnail300KURL") or info.get("OriginalURL", ""), "label_source": "human_positive"})
            if index % 250000 == 0:
                write_json(data.META / "expanded_inventory_progress.json", {"split": split, "phase": "metadata", "rows": index, "matched": len(rows)})
            if len(rows) == len(targets):
                break
        write_csv(output, rows, FIELDS)
        print(f"{split}: eligible {dict(collections.Counter(r['class'] for r in rows if r['eligible']=='1'))}", flush=True)
    inventory_report()


def machine_inventory(threshold=.5):
    """Weak labels are discovery hints ONLY; human negative labels override them."""
    output = data.META / "expanded_oi_machine_train.csv"
    if output.exists():
        inventory_report()
        return
    cache = data.META / "machine_label_targets_train.json"
    if cache.exists():
        selected = json.loads(cache.read_text(encoding="utf-8"))
    else:
        targets = collections.defaultdict(dict)
        url = "https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-machine-imagelabels.csv"
        mids = b"|".join(re.escape(mid.encode()) for mid in CLASS_IDS)
        pattern = re.compile(rb"(?m)^([0-9a-f]{16}),machine,(" + mids + rb"),([0-9.]+)\r?$")
        # This CSV has four unquoted scalar columns. Filter bytes before CSV parsing
        # to avoid allocating dictionaries for ~200M unrelated labels.
        with requests.get(url, headers=data.HEADERS, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            leftover, byte_count, last_report = b"", 0, 0
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                byte_count += len(chunk)
                content = leftover + chunk
                boundary = content.rfind(b"\n")
                if boundary < 0:
                    leftover = content
                    continue
                complete, leftover = content[:boundary + 1], content[boundary + 1:]
                for match in pattern.finditer(complete):
                    identifier, mid, confidence = match.groups()
                    if float(confidence) >= threshold:
                        targets[identifier.decode()][CLASS_IDS[mid.decode()]] = float(confidence)
                if byte_count - last_report >= 100 * 1024 * 1024:
                    last_report = byte_count
                    counts = collections.Counter(fruit for values in targets.values() for fruit in values)
                    write_json(data.META / "machine_inventory_progress.json", {"phase": "labels", "bytes": byte_count, "counts": dict(counts)})
                    print(f"Machine labels {byte_count / 2**20:.0f} MiB: {dict(counts)}", flush=True)
            for match in pattern.finditer(leftover):
                identifier, mid, confidence = match.groups()
                if float(confidence) >= threshold:
                    targets[identifier.decode()][CLASS_IDS[mid.decode()]] = float(confidence)
        selected = {identifier: {"class": next(iter(values)), "confidence": next(iter(values.values()))} for identifier, values in targets.items() if len(values) == 1}
        write_json(cache, selected)
    veto_cache = data.META / "machine_human_veto_train.json"
    if veto_cache.exists():
        veto = set(json.loads(veto_cache.read_text(encoding="utf-8")))
    else:
        veto = set()
        for index, label in enumerate(data.remote_csv(LABEL_URLS["train"]), 1):
            item = selected.get(label["ImageID"])
            fruit = CLASS_IDS.get(label["LabelName"])
            if item and fruit and label.get("Source") in ("verification", "crowdsource-verification"):
                if (fruit == item["class"] and label["Confidence"] == "0") or (fruit != item["class"] and label["Confidence"] == "1"):
                    veto.add(label["ImageID"])
            if index % 2000000 == 0:
                write_json(data.META / "machine_inventory_progress.json", {"phase": "human_veto", "rows": index, "excluded": len(veto)})
        write_json(veto_cache, sorted(veto))
    selected = {identifier: item for identifier, item in selected.items() if identifier not in veto}
    print('Machine single-target candidates after human veto:', dict(collections.Counter(r['class'] for r in selected.values())), flush=True)
    rows = []
    for index, info in enumerate(data.remote_csv(INFO_URLS["train"]), 1):
        identifier = info["ImageID"]
        item = selected.get(identifier)
        if item:
            license_url = info.get("License", "")
            eligible = license_url.rstrip("/") in ("https://creativecommons.org/licenses/by/2.0", "http://creativecommons.org/licenses/by/2.0") and bool(info.get("Author") and info.get("OriginalLandingURL"))
            rows.append({"id": f"oi_{identifier}", "class": item["class"], "source_split": "train", "source_id": identifier,
                         "group_id": f"oi_{identifier}", "xmin": "0", "xmax": "1", "ymin": "0", "ymax": "1", "box_area": "1",
                         "source_url": info.get("OriginalURL", ""), "landing_url": info.get("OriginalLandingURL", ""), "license": license_url,
                         "author": info.get("Author", ""), "title": info.get("Title", ""), "rotation": info.get("Rotation", ""),
                         "eligible": "1" if eligible else "0", "reason": "WEAK_MACHINE_LABEL_requires_visual_and_license_review",
                         "provider": "openimages", "download_url": info.get("Thumbnail300KURL") or info.get("OriginalURL", ""),
                         "label_source": "machine_NOT_ground_truth", "label_confidence": item["confidence"]})
        if index % 250000 == 0:
            write_json(data.META / "machine_inventory_progress.json", {"phase": "metadata", "rows": index, "matched": len(rows)})
        if len(rows) == len(selected):
            break
    write_csv(output, rows, FIELDS)
    write_json(data.META / "machine_inventory_settings.json", {"threshold": threshold, "human_negative_veto": True, "warning": "NOT ground truth; manual review mandatory"})
    inventory_report()


def api(params):
    for attempt in range(6):
        try:
            response = requests.get(COMMONS_API, params=dict(params, action="query", format="json", maxlag=5), headers=data.HEADERS, timeout=(20, 90))
            if response.status_code in (429, 503):
                delay = max(30, int(response.headers.get("Retry-After", "60")))
                print(f"Commons rate limit: wait {delay}s", flush=True)
                time.sleep(delay)
            response.raise_for_status()
            result = response.json()
            if "error" in result:
                raise ValueError(str(result["error"]))
            time.sleep(2)
            return result
        except (requests.RequestException, ValueError):
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)


def plain(value):
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def commons_row(page, fruit):
    info = page.get("imageinfo", [{}])[0]
    ext = info.get("extmetadata", {})
    value = lambda key: plain(ext.get(key, {}).get("value", ""))
    license_url = value("LicenseUrl")
    permitted = bool(re.match(r"^https?://creativecommons\.org/(licenses/(by|by-sa)/[1-4]\.0|publicdomain/(zero/1\.0|mark/1\.0))/?$", license_url))
    eligible = permitted and bool(value("Artist")) and min(info.get("width", 0), info.get("height", 0)) >= 224 and info.get("mime") == "image/jpeg"
    return {"id": f"wc_{page['pageid']}", "class": fruit, "source_split": "commons", "source_id": str(page["pageid"]),
            "group_id": f"wc_{page['pageid']}", "xmin": "0", "xmax": "1", "ymin": "0", "ymax": "1", "box_area": "1",
            "source_url": info.get("url", ""), "landing_url": info.get("descriptionurl", ""), "license": license_url,
            "author": value("Artist"), "title": page["title"], "rotation": "0", "eligible": "1" if eligible else "0",
            "reason": "category_candidate_pending_visual_license_review", "provider": "commons",
            "download_url": info.get("thumburl") or info.get("url", ""), "label_source": "commons_category"}


def commons_inventory(depth=3, max_categories=200, fruits=None):
    for fruit in fruits or ("pineapple", "banana", "strawberry", "orange", "apple"):
        category = CATEGORIES[fruit]
        output = data.META / f"expanded_commons_{fruit}.csv"
        checkpoint = data.META / f"commons_progress_{fruit}.json"
        if output.exists():
            continue
        state = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {"queue": [[f"Category:{category}", 0]], "visited": [], "files": []}
        queue, visited, files = state["queue"], set(state["visited"]), set(state["files"])
        while queue and len(visited) < max_categories:
            name, level = queue[0]
            params = {"list": "categorymembers", "cmtitle": name, "cmlimit": 500, "cmtype": "file|subcat"}
            while True:
                result = api(params)
                for member in result.get("query", {}).get("categorymembers", []):
                    if member["ns"] == 6:
                        files.add(member["pageid"])
                    elif level < depth and member["title"] not in visited and not any(q[0] == member["title"] for q in queue):
                        # Avoid obvious non-photo/non-fruit categories, still review every result.
                        if not any(word in member["title"].lower() for word in (" art", "logos", "symbols", "juice", "cakes", "products", "maps", "diagrams")):
                            queue.append([member["title"], level + 1])
                if "continue" not in result:
                    break
                params.update(result["continue"])
            visited.add(name)
            queue.pop(0)
            write_json(checkpoint, {"queue": queue, "visited": sorted(visited), "files": sorted(files)})
            print(f"Commons {fruit}: {len(visited)} categories, {len(files)} files", flush=True)
        rows = []
        identifiers = sorted(files)
        for offset in range(0, len(identifiers), 50):
            result = api({"pageids": "|".join(map(str, identifiers[offset:offset + 50])), "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 1024})
            rows.extend(commons_row(page, fruit) for page in result.get("query", {}).get("pages", {}).values() if "imageinfo" in page)
        write_csv(output, rows, FIELDS)
        print(f"Commons {fruit}: eligible {sum(r['eligible']=='1' for r in rows)}", flush=True)
    inventory_report()


def candidates():
    by_id = collections.defaultdict(list)
    for path in sorted(data.META.glob("expanded_*.csv")):
        for row in read_csv(path):
            if row["eligible"] == "1":
                by_id[row["id"]].append(row)
    for path in sorted(data.META.glob("candidates_*.csv")):
        for row in read_csv(path):
            if row["eligible"] == "1":
                row.update({"provider": "openimages", "label_source": "human_bbox",
                            "download_url": f"https://open-images-dataset.s3.amazonaws.com/{row['source_split']}/{row['source_id']}.jpg"})
                by_id[row["id"]].append(row)
    rank = {"human_bbox": 0, "human_positive": 1, "commons_category": 2, "machine_NOT_ground_truth": 3}
    return [min(rows, key=lambda row: rank.get(row.get("label_source"), 3)) for rows in by_id.values() if len({r["class"] for r in rows}) == 1]


def inventory_report():
    rows = candidates()
    summary = {fruit: dict(collections.Counter(r["provider"] for r in rows if r["class"] == fruit)) for fruit in CLASSES}
    write_json(data.META / "expanded_inventory_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def download_one(row):
    if row.get("label_source") == "human_bbox":
        # CVDF mirror is already rotated; use bbox crop without reapplying rotation.
        return data.download_one(row)
    result = dict(row)
    original = data.DATA / "raw" / row["provider"] / row["source_split"] / f"{row['source_id']}.jpg"
    crop = data.DATA / "crops" / row["class"] / f"{row['id']}.png"
    try:
        if shutil.disk_usage(ROOT).free < 8 * 2 ** 30:
            raise OSError("Less than 8 GiB free; stop acquisition to protect existing data")
        if not original.exists():
            data.fetch_file(row["download_url"], original)
        with Image.open(original) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        if row["provider"] == "openimages" and row.get("rotation") in ("90", "180", "270"):
            image = image.rotate(int(row["rotation"]), expand=True)
        result.update({"width": image.width, "height": image.height})
        if min(image.size) < 224:
            raise ValueError("Source image too small for native Mid224")
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        if min(image.size) < 224:
            raise ValueError("Extreme aspect ratio leaves insufficient resolution")
        crop.parent.mkdir(parents=True, exist_ok=True)
        image.save(crop, format="PNG")
        result.update({"status": "pending_review", "error": "", "original_path": original.relative_to(ROOT).as_posix(),
                       "crop_path": crop.relative_to(ROOT).as_posix(), "crop_width": image.width, "crop_height": image.height,
                       "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(), "sha256": hashlib.sha256(crop.read_bytes()).hexdigest(), "dhash": data.dhash(image)})
    except Exception as error:
        result.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
    return result


def download(per_class=6500, workers=4, retry_failed=False):
    """One supplemental downloader may update downloads.csv at a time."""
    lock = data.META / "expanded_download.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise ValueError(f"Downloader lock exists: {lock}. Do not run concurrent downloads; remove a stale lock only after checking its PID.") from error
    try:
        with handle:
            handle.write(str(os.getpid()))
            handle.flush()
            return download_batches(per_class, workers, retry_failed)
    finally:
        lock.unlink(missing_ok=True)


def download_batches(per_class, workers, retry_failed):
    path = data.META / "downloads.csv"
    existing = {r["id"]: r for r in read_csv(path)} if path.exists() else {}
    reviews_path = data.META / "reviews.csv"
    rejected = {r["id"] for r in read_csv(reviews_path) if r["decision"] == "reject"} if reviews_path.exists() else set()
    rng = random.Random(2026)
    available = candidates()
    for fruit in CLASSES:
        current = sum(r["class"] == fruit and r["status"] == "pending_review" and r["id"] not in rejected for r in existing.values())
        choices = [r for r in available if r["class"] == fruit and r["id"] not in rejected and (r["id"] not in existing or (retry_failed and existing[r["id"]]["status"] == "failed" and existing[r["id"]].get("provider")))]
        rng.shuffle(choices)
        # Download stronger label sources first, high-confidence machine hints last.
        rank = {"human_bbox": 0, "human_positive": 1, "commons_category": 2, "machine_NOT_ground_truth": 3}
        choices.sort(key=lambda r: (rank.get(r.get("label_source"), 3), -float(r.get("label_confidence") or 0)))
        # Batches limit work already queued once the download target is reached.
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            offset = 0
            while offset < len(choices):
                if current >= per_class:
                    break
                if shutil.disk_usage(ROOT).free < 8 * 2 ** 30:
                    raise OSError("Acquisition stopped: less than 8 GiB free")
                batch = choices[offset:offset + min(workers * 10, per_class - current)]
                futures = [pool.submit(download_one, row) for row in batch]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    existing[result["id"]] = result
                    current += int(result["status"] == "pending_review")
                write_csv(path, sorted(existing.values(), key=lambda r: r["id"]), DOWNLOAD_FIELDS)
                offset += len(batch)
                print(f"{fruit}: {current}/{per_class} downloaded pending candidates; attempted {offset}/{len(choices)}", flush=True)
                acquisition_report(existing.values())
    return acquisition_report(existing.values())


def acquisition_report(rows):
    rows = list(rows)
    result = {fruit: dict(collections.Counter(r["status"] for r in rows if r["class"] == fruit)) for fruit in CLASSES}
    write_json(data.META / "download_summary.json", result)
    reviews_path = data.META / "reviews.csv"
    reviews = {row["id"]: row["decision"] for row in read_csv(reviews_path)} if reviews_path.exists() else {}
    review_counts = {}
    for fruit in CLASSES:
        downloaded = [row for row in rows if row["class"] == fruit and row["status"] == "pending_review"]
        review_counts[fruit] = {"downloaded": len(downloaded), "accepted": sum(reviews.get(r["id"]) == "accept" for r in downloaded),
                                "rejected": sum(reviews.get(r["id"]) == "reject" for r in downloaded),
                                "pending": sum(r["id"] not in reviews for r in downloaded)}
    write_json(data.META / "expanded_download_progress.json", {"classes": result, "reviews": review_counts, "note": "Downloaded candidates, NOT accepted independent samples", "free_gib": shutil.disk_usage(ROOT).free / 2 ** 30})
    return result


def watch(hours=12, per_class=6500, workers=4):
    """Bounded polling: download newly completed inventories, never auto-review."""
    end = time.monotonic() + hours * 3600
    previous = None
    while time.monotonic() < end:
        paths = list(data.META.glob("expanded_*.csv")) + list(data.META.glob("candidates_*.csv"))
        signature = [(p.name, p.stat().st_mtime_ns) for p in sorted(paths)]
        if signature != previous:
            download(per_class, workers)
            previous = signature
            write_json(data.META / "expanded_watcher_progress.json", {"pid": os.getpid(), "state": "waiting_for_more_inventory", "processed": signature, "auto_accept": False})
        time.sleep(60)
    write_json(data.META / "expanded_watcher_progress.json", {"pid": os.getpid(), "state": "time_limit_reached", "auto_accept": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    oi = sub.add_parser("oi-inventory")
    oi.add_argument("--splits", nargs="+", choices=list(LABEL_URLS), default=["validation", "test", "train"])
    machine = sub.add_parser("machine-inventory")
    machine.add_argument("--threshold", type=float, default=.5)
    commons = sub.add_parser("commons-inventory")
    commons.add_argument("--depth", type=int, default=3)
    commons.add_argument("--max-categories", type=int, default=200)
    commons.add_argument("--classes", nargs="+", choices=list(CLASSES))
    dl = sub.add_parser("download")
    dl.add_argument("--per-class", type=int, default=6500)
    dl.add_argument("--workers", type=int, default=4)
    dl.add_argument("--retry-failed", action="store_true")
    watcher = sub.add_parser("watch")
    watcher.add_argument("--hours", type=float, default=12)
    watcher.add_argument("--per-class", type=int, default=6500)
    watcher.add_argument("--workers", type=int, default=4)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.command == "oi-inventory":
        oi_inventory(args.splits)
    elif args.command == "machine-inventory":
        if not .5 <= args.threshold <= 1:
            parser.error("machine threshold must be between .5 and1")
        machine_inventory(args.threshold)
    elif args.command == "commons-inventory":
        if args.depth < 0 or args.max_categories < 1:
            parser.error("depth >=0 and max-categories >=1 required")
        commons_inventory(args.depth, args.max_categories, args.classes)
    elif args.command == "download":
        if args.per_class < 1 or not 1 <= args.workers <= 8:
            parser.error("per-class >=1 and workers between1 and8 required")
        download(args.per_class, args.workers, args.retry_failed)
    elif args.command == "watch":
        if args.hours <= 0 or args.per_class < 1 or not 1 <= args.workers <= 8:
            parser.error("hours >0, per-class >=1, workers between1 and8 required")
        watch(args.hours, args.per_class, args.workers)
    else:
        inventory_report()


if __name__ == "__main__":
    main()