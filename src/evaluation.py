"""Read-only verification of completed runs and single-image inference."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from .common import CLASSES, ROOT, read_csv
from .model import FruitLite
from .training_pipeline import weights_digest
from .training_runtime import classification_metrics, fingerprint


def run_folder(name, root=ROOT):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
        raise ValueError("Invalid run name")
    return Path(root) / "outputs" / name


def read_results(folder, sizes=(224, 32)):
    """Verify saved test receipts without changing files or reading test images."""
    folder = Path(folder)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed":
        raise ValueError("Training has not completed; run train.py to resume")
    manifest = folder / "dataset_manifest.csv"
    if not manifest.exists():
        manifest = folder.parents[1] / "data/metadata/splits.csv"
    rows = read_csv(manifest)
    test_rows = sorted((r for r in rows if r["split"] == "test"), key=lambda r: r["id"])
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    results = {}
    for size in sizes:
        final = folder / f"Mid{size}/final"
        record = summary["results"][str(size)]
        best_path = final / "best.pt"
        best = torch.load(best_path, map_location="cpu", weights_only=True)
        last = torch.load(final / "last.pt", map_location="cpu", weights_only=True)
        measured = json.loads((final / "test/metrics.json").read_text(encoding="utf-8"))
        if (
            best["selection_rule"] != "fixed_final_epoch"
            or last["epoch"] != last["config"]["epochs"]
            or best["epoch"] != last["epoch"]
            or best["identity"] != last["identity"]
            or best["size"] != size
            or last["config"]["size"] != size
            or record["epoch_budget"] != best["epoch"]
        ):
            raise ValueError("Incomplete or incompatible final checkpoints")
        if (
            measured["checkpoint_sha256"] != hashlib.sha256(best_path.read_bytes()).hexdigest()
            or measured["epoch"] != best["epoch"]
            or measured["selected_weights"] != best["selected_weights"]
            or measured["classes"] != list(CLASSES)
            or measured["selection_used_test"] is not False
            or measured["provenance"] != best["provenance"]
            or manifest_sha != best["provenance"]["dataset_manifest_sha256"]
        ):
            raise ValueError("Saved test provenance or checkpoint hash changed")
        expected_identity = fingerprint(
            {
                "run_identity": best["identity"],
                "weights_sha256": weights_digest(best["model"]),
                "epoch": best["epoch"],
                "test_ids": [r["id"] for r in test_rows],
            }
        )
        if measured["evaluation_identity"] != expected_identity:
            raise ValueError("Saved test identity changed")
        predictions = read_csv(final / "test/predictions.csv")
        if (
            not predictions
            or [p["id"] for p in predictions] != [r["id"] for r in test_rows]
            or any(p["true_class"] != r["class"] for p, r in zip(predictions, test_rows))
        ):
            raise ValueError("Test prediction identities or labels changed")
        scores = np.array([[float(p[f"p_{c}"]) for c in CLASSES] for p in predictions])
        if (
            not np.isfinite(scores).all()
            or (scores < 0).any()
            or (scores > 1).any()
            or not np.allclose(scores.sum(1), 1, atol=1e-6, rtol=0)
            or any(
                p["predicted_class"] != CLASSES[int(s.argmax())]
                or not np.isclose(float(p["confidence"]), s.max(), atol=1e-7, rtol=0)
                for p, s in zip(predictions, scores)
            )
        ):
            raise ValueError("Test prediction probabilities changed")
        recomputed = classification_metrics([CLASSES.index(r["class"]) for r in test_rows], scores)
        for key, value in recomputed.items():
            saved = measured["metrics"][key]
            equal = (
                value == saved
                if isinstance(value, (list, dict)) or value is None
                else np.isclose(value, saved, atol=1e-10, rtol=0)
            )
            if not equal:
                raise ValueError(f"Saved test metric changed: {key}")
        if record["test_accuracy"] != recomputed["accuracy"] or record["test_macro_f1"] != recomputed["macro_f1"]:
            raise ValueError("Run summary does not match test predictions")
        results[str(size)] = measured
    return results


def print_results(results):
    for size, record in results.items():
        m = record["metrics"]
        print(
            f"SAVED TEST Mid{size}: {m['correct']}/{m['total']} "
            f"accuracy={m['accuracy']:.2%} macro_f1={m['macro_f1']:.6f} "
            f"epoch={record['epoch']} weights={record['selected_weights']}",
            flush=True,
        )


def predict_image(folder, image_path, size=224, device="auto"):
    read_results(folder, sizes=(size,))
    checkpoint = torch.load(Path(folder) / f"Mid{size}/final/best.pt", map_location="cpu", weights_only=True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FruitLite(**checkpoint["model_settings"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(tensor).float().softmax(1)[0].cpu().tolist()
    return dict(sorted(zip(CLASSES, probabilities), key=lambda item: item[1], reverse=True))
