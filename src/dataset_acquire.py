"""Resumable real-photo dataset candidates; quantity first, never auto-accept.

Uses cached public Kaggle inventories in data/metadata/sources. Outputs are
staging PNGs, NOT official Mid224/Mid32: production byte gates and review/split
gates remain unchanged.
"""

import argparse
import collections
import concurrent.futures
import hashlib
import io
import json
import os
import random
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image, ImageOps

from . import compression, data, dataset_review
from .common import CLASSES, ROOT, write_json
from .dataset_quota import apply_retirements

META = ROOT / "data" / "metadata" / "fruit_acquisition_08"
OUT = ROOT / "data" / "fruit_sources_08"
SOURCES = (
    ("07", "mohammedarfathr/orange-fruit-daatset", "orange"),
    ("07", "adhilpk/pineapple", "pineapple"),
    ("07", "chrisfilo/fruit-recognition", None),
    ("07", "utkarshsaxenadn/fruits-classification", None),
    ("07", "shashwatwork/fruitnet-indian-fruits-dataset-with-quality", None),
    ("08", "mahyeks/multi-class-strawberry-ripeness-detection-dataset", "strawberry"),
)
MAX_FILE_BYTES = 12_000_000


def identify(name, fixed=None):
    """Only known upstream class folders, never substring-match fruit names."""
    path = Path(name)
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return None
    lowered = name.lower()
    if any(hint in lowered for hint in ("edited", "augment", "total number", "synthetic")):
        return None
    if fixed:
        return fixed
    for part in reversed(path.parts[:-1]):
        label = part.lower()
        if label.endswith(("_good", "_bad")):
            label = label.rsplit("_", 1)[0]
        if label in CLASSES:
            return label
    return None


def candidates():
    rows = []
    notices = []
    for phase, ref, fixed in SOURCES:
        folder = (ROOT / "data" / "metadata" / "sources"
                  / (phase + "_source_metadata") / ref.replace("/", "__"))
        state = json.loads((folder / "inventory.json").read_text(encoding="utf-8"))
        view = json.loads((folder / "view.json").read_text(encoding="utf-8"))
        if not state["complete"]:
            raise ValueError(f"Incomplete inventory: {ref}")
        notices.append({"dataset_ref": ref, "license_declared": view.get("licenseName"),
                        "metadata_path": str(folder.relative_to(ROOT)),
                        "landing_url": "https://www.kaggle.com/datasets/" + ref,
                        "rights_verified": False})
        for item in state["rows"]:
            fruit = identify(item["name"], fixed)
            if not fruit or not 0 < item["totalBytes"] <= MAX_FILE_BYTES:
                continue
            identity = ref + "/" + item["name"]
            rows.append({"id": "kaggle_" + hashlib.sha256(identity.encode()).hexdigest()[:20],
                         "class": fruit, "dataset_ref": ref, "source_identity": identity,
                         "source_name": item["name"], "declared_bytes": item["totalBytes"],
                         "license_declared": view.get("licenseName", "Unknown"),
                         "group_id": "kaggle_dataset_" + ref.replace("/", "_"),
                         "group_note": "Conservative dataset group; physical sessions unverified",
                         "download_url": "https://www.kaggle.com/api/v1/datasets/download/" + ref + "/" + urllib.parse.quote(item["name"], safe="")})
    extra = META / "additional_candidates.json"
    if extra.exists():
        additions = json.loads(extra.read_text(encoding="utf-8"))
        for row in additions:
            if (row["class"] not in CLASSES or not 0 < row["declared_bytes"] <= MAX_FILE_BYTES
                    or row.get("license_declared") in (None, "Unknown")
                    or not permitted_candidate_url(row)):
                raise ValueError("Invalid additional candidate")
        rows.extend(additions)
        extra_notices = META / "additional_source_notices.json"
        if not extra_notices.exists():
            raise ValueError("Additional candidate provenance notices missing")
        notices.extend(json.loads(extra_notices.read_text(encoding="utf-8")))
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source identities")
    return rows, notices


def permitted_candidate_url(row):
    """Permit only existing Kaggle URLs or provenance-backed official OI images."""
    url = row.get("download_url", "")
    if url.startswith("https://www.kaggle.com/api/v1/datasets/download/"):
        return True
    if row.get("provider") == "mendeley_archive":
        approved_archives = {
            "10.17632/tv7zfd6yzz.1": "https://data.mendeley.com/public-files/datasets/tv7zfd6yzz/files/8b6320ff-0d8f-47f5-8cfb-58a00d515b51/file_downloaded",
            "10.17632/xcp2sj2jkp.1": "https://data.mendeley.com/public-files/datasets/xcp2sj2jkp/files/b23bd73c-b6c5-4585-8970-0ead829d3724/file_downloaded",
            "10.17632/yjydjzww2j.3": "https://data.mendeley.com/public-files/datasets/yjydjzww2j/files/3b02b8bb-8044-4422-b701-ed3bf6da31ad/file_downloaded",
        }
        member = row.get("archive_member", "")
        doi = row.get("source_doi", "")
        identity = doi + "/" + member
        digest = row.get("source_sha256", "")
        original_member = (
            doi == "10.17632/tv7zfd6yzz.1" and member.startswith("PineappleLeaf-Fruit Dataset/Raw Dataset/Fruits/")
            or doi == "10.17632/xcp2sj2jkp.1" and member.startswith("train/images/")
            or doi == "10.17632/yjydjzww2j.3" and member.startswith("PinesFOPT-OPT/")
        )
        return (doi in approved_archives and url == approved_archives[doi]
                and bool(member) and not Path(member).is_absolute() and ".." not in Path(member).parts
                and original_member and row.get("class") == "pineapple"
                and Path(member).suffix.lower() in {".jpg", ".jpeg", ".png"}
                and row.get("source_identity") == identity
                and row.get("id") == "mendeley_" + hashlib.sha256(identity.encode()).hexdigest()[:20]
                and row.get("license_declared") == "https://creativecommons.org/licenses/by/4.0/"
                and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
                and bool(row.get("author") and row.get("landing_url"))
                and len(row.get("archive_sha256", "")) == 64)
    if row.get("provider") != "openimages":
        return False
    source_id = row.get("source_id", "")
    split = row.get("source_split", "")
    if (len(source_id) != 16 or any(c not in "0123456789abcdef" for c in source_id)
            or split not in {"train", "validation", "test"}):
        return False
    license_url = row.get("license_declared", "").rstrip("/")
    return (url == f"https://open-images-dataset.s3.amazonaws.com/{split}/{source_id}.jpg"
            and row.get("id") == "oi_" + source_id
            and row.get("source_identity") == "openimages/" + source_id
            and bool(row.get("author") and row.get("landing_url") and row.get("source_url"))
            and license_url in {"https://creativecommons.org/licenses/by/2.0", "http://creativecommons.org/licenses/by/2.0"}
            and row.get("label_evidence") == "human_positive")


def prepare(opened):
    if opened.width * opened.height > 40_000_000 or max(opened.size) > 8000:
        raise ValueError("Native image exceeds pixel safety cap")
    if getattr(opened, "n_frames", 1) != 1:
        raise ValueError("Animated/multiframe source")
    native = ImageOps.exif_transpose(opened).convert("RGB")
    native.load()
    if min(native.size) < 224:
        raise ValueError("Native image below224; no upscaling")
    square = ImageOps.pad(native, (max(native.size),) * 2,
                          method=Image.Resampling.LANCZOS, color=(127, 127, 127))
    pairs = {size: square.resize((size, size), Image.Resampling.LANCZOS) for size in (224, 32)}
    return native, pairs


def read_bounded(response):
    declared = response.headers.get("Content-Length")
    if declared and int(declared) > MAX_FILE_BYTES:
        raise ValueError("Declared download exceeds per-file safety cap")
    blob = response.read(MAX_FILE_BYTES + 1)
    if len(blob) > MAX_FILE_BYTES:
        raise ValueError("Download exceeds per-file safety cap")
    return blob


def atomic_bytes(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(path)


def unwrap_image(blob, source_name):
    """Some single-file Kaggle responses wrap the requested image in a ZIP."""
    if not blob.startswith(b"PK\x03\x04"):
        return blob
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        members = archive.infolist()
        if (len(members) != 1 or members[0].is_dir()
                or Path(members[0].filename).name != Path(source_name).name
                or members[0].file_size > MAX_FILE_BYTES
                or members[0].flag_bits & 1):
            raise ValueError("Unexpected or oversized single-image ZIP")
        with archive.open(members[0]) as stream:
            payload = stream.read(MAX_FILE_BYTES + 1)
        if len(payload) > MAX_FILE_BYTES:
            raise ValueError("Expanded image exceeds safety cap")
        return payload


def fetch_candidate(row, local_sources=None):
    result = dict(row, status="failed", transferred_bytes=0, outputs={})
    original = OUT / "originals" / row["class"] / (row["id"] + Path(row["source_name"]).suffix.lower())
    try:
        existing = (local_sources or {}).get(row["id"])
        if original.exists():
            blob = original.read_bytes()
            result["reused_local"] = True
        elif existing:
            source = ROOT / existing["local_path"]
            blob = source.read_bytes()
            if hashlib.sha256(blob).hexdigest() != existing["original_sha256"]:
                raise ValueError("Existing pilot original hash mismatch")
            # Reference preserved pilot source without copying/downloading again.
            original = source
            result["reused_pilot"] = True
        else:
            with urllib.request.urlopen(row["download_url"], timeout=45) as response:
                blob = read_bounded(response)
            result["transferred_bytes"] = len(blob)
            blob = unwrap_image(blob, row["source_name"])
            atomic_bytes(original, blob)
        if row.get("source_sha256") and hashlib.sha256(blob).hexdigest() != row["source_sha256"]:
            raise ValueError("Registered source hash mismatch")
        result.update(original_path=original.relative_to(ROOT).as_posix(),
                      original_sha256=hashlib.sha256(blob).hexdigest(), original_bytes=len(blob))
        with Image.open(io.BytesIO(blob)) as opened:
            native, images = prepare(opened)
        result.update(native_size=list(native.size),
                      native_pixel_sha256=hashlib.sha256(str(native.size).encode() + native.tobytes()).hexdigest(),
                      dhash=data.dhash(native),
                      pixel224_sha256=hashlib.sha256(images[224].tobytes()).hexdigest())
        for size, image in images.items():
            payload, _, rmse = compression.encode_image(image, "PNG", {"optimize": True})
            if rmse:
                raise ValueError("PNG pixels changed")
            destination = OUT / "paired" / str(size) / row["class"] / (row["id"] + ".png")
            atomic_bytes(destination, payload)
            result["outputs"][str(size)] = {"path": destination.relative_to(ROOT).as_posix(),
                                            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        result.update(status="pending_review", label_source=row.get("label_evidence", "upstream_folder_hint"),
                      photo_verified=False, rights_verified=False, physical_groups_verified=False,
                      development_only=data.reserved_for_development(result, data.development_reservations()))
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, urllib.error.HTTPError):
            result["http_status"] = error.code
    return result


def load_records(path):
    """Repair only an interrupted final journal write; never hide older corruption."""
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    records = []
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if i != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError("Corrupted acquisition journal")
            # This file belongs solely to this worker; preserve damaged tail separately.
            path.with_suffix(".interrupted_tail").write_bytes(line)
            path.write_bytes(b"".join(lines[:i]))
    return apply_retirements(records, path.parent / "quota_retirements.json")


def validate_existing(records):
    for row in records:
        if row["status"] not in {"pending_review", "duplicate_exact"}:
            continue
        for item in [{"path": row["original_path"], "sha256": row["original_sha256"]}, *row["outputs"].values()]:
            path = ROOT / item["path"]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Changed/missing persisted image: {path}")


def mark_duplicate(row, hashes):
    if row["status"] != "pending_review":
        return
    keys = [(field, row[field]) for field in ("original_sha256", "native_pixel_sha256", "pixel224_sha256")]
    matches = {hashes[key] for key in keys if key in hashes}
    if matches:
        row.update(status="duplicate_exact", duplicate_of=sorted(matches))
    for key in keys:
        hashes.setdefault(key, row["id"])


def summary(records, available, reason, blocked, target=5000):
    counts = {fruit: dict(collections.Counter(r["status"] for r in records if r["class"] == fruit)) for fruit in CLASSES}
    reviewed = dataset_review.review_summary(records, dataset_review.load_approval(META / "user_approval.json"))
    return {"pid": os.getpid(), "updated_unix": time.time(), "stop_reason": reason,
            "counts": counts, "inventory_candidates": dict(collections.Counter(r["class"] for r in available)),
            "target_per_class": target,
            "pending_shortfall_to_target": {c: max(0, target - counts[c].get("pending_review", 0)) for c in CLASSES},
            "pending_shortfall_to_5050": {c: max(0, 5050 - counts[c].get("pending_review", 0)) for c in CLASSES},
            "accepted_images": reviewed["accepted_images"], "review_complete": False, "official_export": False,
            "accepted_shortfall_to_target": {c: max(0, target - reviewed["accepted_per_class"][c]) for c in CLASSES},
            "review_counts": reviewed["review_counts"],
            "accepted_shortfall_to_5050": reviewed["accepted_shortfall_to_5050"],
            "approval_basis": reviewed["approval_basis"],
            "transferred_bytes": sum(r.get("transferred_bytes", 0) for r in records),
            "blocked_sources": sorted(blocked), "production_limits_unchanged": data.DATASET_LIMIT_BYTES,
            "quantity_first": True, "staging_may_exceed_production_limits": True}


def run(args):
    META.mkdir(parents=True, exist_ok=True)
    lock = META / "worker.lock"
    # Do not remove stale locks automatically: inspect recorded PID before recovery.
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w") as handle:
        json.dump({"pid": os.getpid(), "started_unix": time.time()}, handle)
    try:
        rows, notices = candidates()
        write_json(META / "source_notices.json", notices)
        write_json(META / "candidates.json", rows)
        journal = META / "attempts.jsonl"
        records = load_records(journal)
        validate_existing(records)
        known = {r["id"] for r in records}
        blocked = {r["dataset_ref"] for r in records if r.get("http_status") in (401, 403, 429)}
        hashes = {}
        for row in records:
            if row["status"] in {"pending_review", "duplicate_exact", "retired_quota"}:
                for field in ("original_sha256", "native_pixel_sha256", "pixel224_sha256"):
                    hashes.setdefault((field, row[field]), row["id"])
        pilot = ROOT / "data" / "metadata" / "compression_study_07_fruit" / "manifest.json"
        local = {r["id"]: r for r in json.loads(pilot.read_text(encoding="utf-8"))} if pilot.exists() else {}
        strata = collections.defaultdict(list)
        rng = random.Random(2026)
        for row in rows:
            if row["id"] not in known:
                strata[(row["class"], row["dataset_ref"])].append(row)
        for items in strata.values():
            rng.shuffle(items)
        rotation = collections.deque(sorted(strata))
        start = time.monotonic()
        transferred = 0
        attempts = 0
        reason = "running"
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool, journal.open("a", encoding="utf-8", newline="\n") as handle:
            while True:
                if (META / "STOP").exists():
                    reason = "stop_file"
                    break
                if time.monotonic() - start >= args.hours * 3600:
                    reason = "time_safety_cap"
                    break
                if attempts >= args.max_attempts:
                    reason = "attempt_batch_cap"
                    break
                if shutil.disk_usage(ROOT).free < 5_000_000_000:
                    reason = "disk_reserve_5GB"
                    break
                pending = collections.Counter(r["class"] for r in records if r["status"] == "pending_review")
                batch = []
                batch_limit = min(args.workers, args.max_attempts - attempts)
                while len(batch) < batch_limit:
                    before = len(batch)
                    for _ in range(len(rotation)):
                        key = rotation[0]
                        rotation.rotate(-1)
                        fruit, ref = key
                        if ref not in blocked and pending[fruit] < args.per_class and strata[key]:
                            batch.append(strata[key].pop())
                            pending[fruit] += 1
                            if len(batch) >= batch_limit:
                                break
                    if len(batch) == before:
                        break  # All remaining strata exhausted, blocked or at quota.
                if not batch:
                    reason = "target_or_inventory_exhausted"
                    break
                if transferred + len(batch) * MAX_FILE_BYTES > args.max_transfer_gb * 1_000_000_000:
                    reason = "transfer_safety_cap"
                    break
                for result in pool.map(lambda row: fetch_candidate(row, local), batch):
                    mark_duplicate(result, hashes)
                    records.append(result)
                    attempts += 1
                    transferred += result.get("transferred_bytes", 0)
                    if result.get("http_status") in (401, 403, 429):
                        blocked.add(result["dataset_ref"])
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                state = summary(records, rows, "running", blocked, args.per_class)
                write_json(META / "progress.json", state)
                if attempts % 25 < args.workers:
                    print(json.dumps({"attempted_this_run": attempts, "counts": state["counts"], "transfer_bytes_this_run": transferred}), flush=True)
                time.sleep(.2)
        state = summary(records, rows, reason, blocked, args.per_class)
        write_json(META / "progress.json", state)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
        return state
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=5000, help="Active technical candidates, not accepted quota")
    parser.add_argument("--max-attempts", type=int, default=40000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--hours", type=float, default=12)
    parser.add_argument("--max-transfer-gb", type=float, default=25)
    args = parser.parse_args()
    if not (1 <= args.per_class <= 10000 and 1 <= args.max_attempts <= 60000 and 1 <= args.workers <= 4
            and 0 < args.hours <= 24 and 0 < args.max_transfer_gb <= 40):
        parser.error("Invalid bounded acquisition options")
    run(args)


if __name__ == "__main__":
    main()
