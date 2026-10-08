"""Read-only candidate sampling and codec feasibility study, never dataset acceptance."""

import argparse
import collections
import hashlib
import io
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps, features

from . import data
from .common import CLASSES, ROOT, read_csv, write_csv, write_json


def sample_candidates(rows, reviews, per_class=100, seed=2026):
    if per_class < 1:
        raise ValueError("per-class must be positive")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate IDs")
    rng = random.Random(seed)
    selected = []
    for fruit in CLASSES:
        strata = collections.defaultdict(list)
        for row in sorted(rows, key=lambda r: r["id"]):
            if row["class"] == fruit and row["status"] == "pending_review" and reviews.get(row["id"]) != "reject":
                strata[(row.get("provider", ""), row.get("label_source", ""))].append(row)
        available = sum(map(len, strata.values()))
        count = min(per_class, available)
        if not count:
            continue
        allocations = {key: count * len(values) // available for key, values in strata.items()}
        ranked = sorted(strata, key=lambda key: (-(count * len(strata[key]) % available), key))
        for key in ranked[:count - sum(allocations.values())]:
            allocations[key] += 1
        for key in sorted(strata):
            rng.shuffle(strata[key])
            selected.extend(dict(row) for row in strata[key][:allocations[key]])
    return selected


def codec_configs():
    configs = [("png", "PNG", {"optimize": True})]
    for quality in (0, 1, 5, 10, 30):
        configs.append((f"jpeg_q{quality}", "JPEG", {"quality": quality, "optimize": True}))
        if features.check("webp"):
            configs.append((f"webp_q{quality}", "WEBP", {"quality": quality, "method": 4}))
    return configs


def encode_image(image, codec, settings):
    buffer = io.BytesIO()
    image.save(buffer, format=codec, **settings)
    payload = buffer.getvalue()
    with Image.open(io.BytesIO(payload)) as opened:
        decoded = opened.convert("RGB")
        decoded.load()
    if decoded.size != image.size:
        raise ValueError("Codec changed dimensions")
    difference = np.asarray(image, dtype=np.float32) - np.asarray(decoded, dtype=np.float32)
    return payload, decoded, float(np.sqrt(np.mean(difference ** 2)))


def summarize(measurements, size):
    values = sorted(row["bytes"] for row in measurements)
    if not values:
        raise ValueError("No measurements")
    mean = sum(values) / len(values)
    full_count = len(CLASSES) * sum(data.SPLIT_QUOTAS.values())
    limit = data.DATASET_LIMIT_BYTES[size]
    return {"sample_images": len(values), "sample_total_bytes": sum(values), "mean_bytes": mean,
            "min_bytes": values[0], "p95_bytes": values[math.ceil(.95 * len(values)) - 1], "max_bytes": values[-1],
            "projected_bytes": round(mean * full_count), "projected_mib": mean * full_count / 1024 ** 2,
            "limit_bytes": limit, "projected_within_limit": mean * full_count <= limit,
            "mean_rgb_rmse": sum(row["rgb_rmse"] for row in measurements) / len(values),
            "by_class": {fruit: {"images": len(items), "mean_bytes": sum(r["bytes"] for r in items) / len(items)}
                         for fruit in CLASSES if (items := [r for r in measurements if r["class"] == fruit])}}


def study(output, per_class=100, seed=2026, preview_per_class=5):
    output = Path(output)
    if not 0 <= preview_per_class <= per_class:
        raise ValueError("preview-per-class must be between 0 and per-class")
    if output.exists():
        raise ValueError("Study output exists; use a new directory. No overwrite allowed.")
    rows = read_csv(data.META / "downloads.csv")
    review_path = data.META / "reviews.csv"
    reviews = {r["id"]: r["decision"] for r in read_csv(review_path)} if review_path.exists() else {}
    sample = sample_candidates(rows, reviews, per_class, seed)
    if not sample:
        raise ValueError("No downloaded non-rejected candidates")
    output.mkdir(parents=True)
    fields = list(dict.fromkeys(key for row in sample for key in row))
    write_csv(output / "sample.csv", sample, fields)
    # Reserve both IDs and known parent groups from future test selection; not acceptance.
    write_json(output / "development_reservations.json", {"reason": "codec selection/visual inspection; exclude these IDs and related groups from held-out test", "ids": [r["id"] for r in sample], "group_ids": sorted({r["group_id"] for r in sample if r.get("group_id")}), "original_sha256": sorted({r["original_sha256"] for r in sample if r.get("original_sha256")}), "sha256": sorted({r["sha256"] for r in sample if r.get("sha256")}), "dhash": sorted({r["dhash"] for r in sample if r.get("dhash")})})
    write_csv(output / "review_template.csv", [{"id": r["id"], "class": r["class"], "label_ok": "", "license_ok": "", "independent_group": "", "mid224_quality_ok": "", "mid32_quality_ok": "", "notes": ""} for r in sample])
    configs = codec_configs()
    measurements = []
    errors = []
    preview_counts = collections.Counter()
    sheets = {}
    columns = [(224, "png"), (224, "jpeg_q1"), (224, "webp_q0" if features.check("webp") else "jpeg_q0"),
               (224, "webp_q5" if features.check("webp") else "jpeg_q5"), (32, "png"), (32, "webp_q30" if features.check("webp") else "jpeg_q30")]
    for index, row in enumerate(sample):
        try:
            path = ROOT / row["crop_path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Crop hash differs from snapshot")
            with Image.open(path) as opened:
                if opened.mode != "RGB" or min(opened.size) < 224:
                    raise ValueError("Invalid source mode/dimensions")
                square = ImageOps.pad(opened, (max(opened.size),) * 2, method=Image.Resampling.LANCZOS, color=(127, 127, 127))
                square.load()
            preview = preview_counts[row["class"]] < preview_per_class
            if preview:
                canvas = sheets.setdefault(row["class"], Image.new("RGB", (len(columns) * 224, preview_per_class * 270), "white"))
                draw = ImageDraw.Draw(canvas)
                y = preview_counts[row["class"]] * 270
            candidate_measurements = []
            for size in data.DATASET_LIMIT_BYTES:
                image = square.resize((size, size), Image.Resampling.LANCZOS)
                for name, codec, settings in configs:
                    payload, decoded, rmse = encode_image(image, codec, settings)
                    candidate_measurements.append({"id": row["id"], "class": row["class"], "size": size,
                                         "configuration": name, "bytes": len(payload), "rgb_rmse": rmse})
                    if preview and (size, name) in columns:
                        x = columns.index((size, name)) * 224
                        canvas.paste(decoded.resize((224, 224), Image.Resampling.NEAREST) if size == 32 else decoded, (x, y))
                        draw.text((x + 2, y + 225), row["id"], fill="black")
                        draw.text((x + 2, y + 240), f"{size} {name} {len(payload)}B", fill="black")
            measurements.extend(candidate_measurements)
            if preview:
                preview_counts[row["class"]] += 1
        except (OSError, ValueError) as error:
            errors.append({"id": row["id"], "error": str(error)})
        if (index + 1) % 50 == 0:
            print(f"Compression study: {index + 1}/{len(sample)} candidates", flush=True)
    for fruit, canvas in sheets.items():
        canvas.save(output / f"comparison_{fruit}.png")  # Lossless sheets, no extra JPEG artifacts.
    if measurements:
        write_csv(output / "measurements.csv", measurements)
    summaries = {}
    for size in data.DATASET_LIMIT_BYTES:
        for name, codec, settings in configs:
            items = [r for r in measurements if r["size"] == size and r["configuration"] == name]
            if items:
                summaries[f"{size}_{name}"] = {"codec": codec, "settings": settings, **summarize(items, size)}
    report = {"seed": seed, "requested_per_class": per_class, "sample_counts": dict(collections.Counter(r["class"] for r in sample)),
              "source_strata": dict(collections.Counter(f"{r['class']}/{r.get('provider', '')}/{r.get('label_source', '')}" for r in sample)),
              "errors": errors, "configurations": summaries, "auto_accept": False,
              "official_dataset_ready": False, "quality_verified": False,
              "note": "Pending-label candidates, stratified by class/provider/label_source. Projections are not guarantees; RMSE does not prove recognition, licensing, independence or acceptable quality. q0/q1 are exploratory boundaries, NOT approved production settings."}
    write_json(output / "report.json", report)
    print(f"Study report: {output / 'report.json'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--preview-per-class", type=int, default=5)
    args = parser.parse_args()
    if args.per_class < 1 or not 0 <= args.preview_per_class <= args.per_class:
        parser.error("per-class >=1 and 0 <= preview-per-class <= per-class required")
    study(args.output, args.per_class, args.seed, args.preview_per_class)


if __name__ == "__main__":
    main()