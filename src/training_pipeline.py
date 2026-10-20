"""One-command development, full-pool refit and locked test of both resolutions."""

from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil

import torch
from torchvision import transforms

from .common import CLASSES, ROOT, read_csv, write_csv, write_json
from .input_audit_cache import validate_inputs
from .model import FruitLite
from .profile import profile
from .train import ManifestDataset, loader, seed_all
from .training_data import development_split
from .training_runtime import RunLog, TrainConfig, exclusive_run, fingerprint, run_epoch, source_digest, train_job


AUGMENTATION = {
    "crop_scale_224": [0.75, 1.0],
    "crop_scale_32": [0.85, 1.0],
    "crop_ratio": [0.9, 1.1],
    "horizontal_flip_probability": 0.5,
    "rotation_degrees": 10,
    "rotation_fill_rgb": [127, 127, 127],
    "color_jitter": {"brightness": 0.15, "contrast": 0.15, "saturation": 0.1, "hue": 0.015},
    "random_erasing_probability": 0.1,
    "random_erasing_scale": [0.02, 0.12],
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225],
    "pretrained_weights": False,
}


def dataset(root, size, rows, training=False):
    result = ManifestDataset(size, "explicit_partition", training, rows=rows, root=root)
    if training:
        result.transform = transforms.Compose(
            [result.transform, transforms.RandomErasing(p=0.1, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0.0)]
        )
    return result


def plot_history(folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = json.loads((folder / "epochs.json").read_text(encoding="utf-8"))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for phase in ("train", "val", "ema_val"):
        values = [entry for entry in history if entry[phase] is not None]
        if values:
            axes[0].plot([e["epoch"] for e in values], [e[phase]["loss"] for e in values], label=phase)
            axes[1].plot([e["epoch"] for e in values], [100 * e[phase]["accuracy"] for e in values], label=phase)
    for axis, title in zip(axes, ("Loss", "Top-1 accuracy (%)")):
        axis.set(xlabel="Completed epoch", ylabel=title)
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(folder / "learning_curves.png", dpi=160)
    plt.close(figure)


def weights_digest(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def evaluate_locked(final_folder, test_data, config):
    checkpoint_path = final_folder / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["selection_rule"] != "fixed_final_epoch":
        raise ValueError("Test requires completed full-pool refit and a locked epoch budget")
    final = torch.load(final_folder / "last.pt", map_location="cpu", weights_only=True)
    if final["epoch"] != final["config"]["epochs"] or checkpoint["epoch"] != final["epoch"]:
        raise ValueError("Cannot evaluate an incomplete final run")
    identity = fingerprint(
        {
            "run_identity": checkpoint["identity"],
            "weights_sha256": weights_digest(checkpoint["model"]),
            "epoch": checkpoint["epoch"],
            "test_ids": [r["id"] for r in test_data.rows],
        }
    )
    output = final_folder / "test"
    metrics_file = output / "metrics.json"
    if metrics_file.exists():
        measured = json.loads(metrics_file.read_text(encoding="utf-8"))
        if measured["evaluation_identity"] != identity:
            raise ValueError("Locked test already exists for different weights/data; preserve it")
        print(f"SKIP test Mid{config.size}: existing result {measured['metrics']['accuracy']:.2%}", flush=True)
        return measured
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda"
        if config.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if config.device == "auto"
        else config.device
    )
    model = FruitLite(**checkpoint["model_settings"]).to(device)
    model.load_state_dict(checkpoint["model"])
    test_loader = loader(test_data, config.batch_size, config.workers, False, config.seed, device.type == "cuda")
    log = RunLog(output)
    try:
        measured, predictions, _ = run_epoch(
            model, test_loader, device, config, log, epoch=checkpoint["epoch"], phase="test"
        )
        records = []
        for row, truth, scores in zip(test_data.rows, predictions["labels"], predictions["probabilities"]):
            prediction = max(range(5), key=lambda i: scores[i])
            records.append(
                {
                    "id": row["id"],
                    "true_class": CLASSES[truth],
                    "predicted_class": CLASSES[prediction],
                    "confidence": scores[prediction],
                    **{f"p_{fruit}": scores[i] for i, fruit in enumerate(CLASSES)},
                }
            )
        write_csv(output / "predictions.csv", records)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(7, 6))
        axis.imshow(measured["confusion_matrix"], cmap="Blues")
        axis.set(
            xticks=range(5),
            yticks=range(5),
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            xlabel="Predicted",
            ylabel="True",
            title=f"Mid{config.size} locked test: {measured['accuracy']:.2%}",
        )
        plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
        for y in range(5):
            for x in range(5):
                axis.text(x, y, str(measured["confusion_matrix"][y][x]), ha="center", va="center")
        figure.tight_layout()
        figure.savefig(output / "confusion_matrix.png", dpi=160)
        plt.close(figure)
        result = {
            "evaluation_identity": identity,
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "epoch": checkpoint["epoch"],
            "selected_weights": checkpoint["selected_weights"],
            "classes": list(CLASSES),
            "metrics": measured,
            "selection_used_test": False,
            "provenance": checkpoint["provenance"],
        }
        # Written last, so a completed test can be recognized without rerunning it.
        write_json(metrics_file, result)
        log.say(f"RESULT Mid{config.size} TEST top1={measured['accuracy']:.2%} macro_f1={measured['macro_f1']:.4f}")
        return result
    finally:
        log.close()


def run_pipeline(
    root=ROOT,
    run_name="fruitlite_l2_seed2026",
    sizes=(224, 32),
    epochs=None,
    batch_size=None,
    workers=None,
    device=None,
    validation_fraction=0.1,
    check_only=False,
    full_audit=False,
):
    root = Path(root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", run_name):
        raise ValueError("Invalid run name: use letters, digits, underscore or hyphen")
    if not sizes or len(set(sizes)) != len(sizes) or any(s not in (224, 32) for s in sizes):
        raise ValueError("Select Mid224 and/or Mid32 once")
    folder = root / "outputs" / run_name
    with exclusive_run(folder / ".pipeline.lock"):
        return _run_pipeline(
            root, folder, sizes, epochs, batch_size, workers, device, validation_fraction, check_only, full_audit
        )


def _run_pipeline(root, folder, sizes, epochs, batch_size, workers, device, fraction, check_only, full_audit):
    configs = {}
    for size in sizes:
        path = root / f"configs/mid{size}.json"
        values = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else asdict(
                TrainConfig(size=size, epochs=180 if size == 224 else 220, batch_size=64 if size == 224 else 256)
            )
        )
        for key, override in (("epochs", epochs), ("batch_size", batch_size), ("workers", workers), ("device", device)):
            if override is not None:
                values[key] = override
        configs[size] = TrainConfig(**values)
        configs[size].validate()
        if configs[size].device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if configs[size].size != size:
            raise ValueError(f"Configuration size does not match Mid{size}")
    if len({c.seed for c in configs.values()}) != 1:
        raise ValueError("Both resolutions must share the frozen partition seed")
    seed = next(iter(configs.values())).seed
    manifest_path = root / "data/metadata/splits.csv"
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    rows = read_csv(manifest_path)
    train_rows, val_rows, partition = development_split(rows, fraction, seed)
    assignments = partition.pop("assignments")
    provenance = {
        "dataset_manifest_sha256": manifest_sha,
        "partition": partition,
        "augmentation": AUGMENTATION,
        "test_used_for_selection": False,
    }
    review_path = root / "data/metadata/build_settings.json"
    if review_path.exists():
        provenance["dataset_review"] = json.loads(review_path.read_text(encoding="utf-8"))
    plan = {
        "configs": {str(s): asdict(c) for s, c in configs.items()},
        "provenance": provenance,
        "source_sha256": source_digest(),
        "stages": ["development", "full_pool_refit", "locked_test"],
    }
    plan_identity = fingerprint(plan)
    plan_path = folder / "workflow.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8"))["identity"] != plan_identity:
        raise ValueError("Workflow settings, code or manifest changed; use a new --run-name")
    write_json(plan_path, {"identity": plan_identity, **plan})
    write_csv(folder / "development_split.csv", assignments)
    print(f"OUTPUT: {folder}\nINPUT CHECK: reusing a decoded audit when valid; verifying image hashes...", flush=True)
    cache_path = root / "data/metadata/input_audit_cache.json"
    with exclusive_run(cache_path.with_suffix(".lock")):
        report, reused = validate_inputs(root, rows, cache_path, force_full=full_audit)
    write_json(
        folder / "input_validation.json",
        {
            **report,
            "cached_decoded_audit_reused": reused,
            "current_verification": "inventory_and_all_encoded_sha256" if reused else "full_decode_and_sha256",
        },
    )
    print(
        "INPUT CHECK PASSED: all image hashes match; decoded audit reused."
        if reused
        else "INPUT AUDIT PASSED: full decoded audit cached for the next restart.",
        flush=True,
    )
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_sha:
        raise ValueError("Manifest changed during validation")
    print(
        f"PARTITION train={len(train_rows)} val={len(val_rows)} test_reserved={partition['test_reserved']}", flush=True
    )
    print(
        f"DATA REVIEW: {partition['unverified_capture_images']} training images have unverified physical groups; "
        "validation can contain unknown near duplicates. Source/label review status is preserved in settings.",
        flush=True,
    )
    seed_all(seed)
    profiles = {str(s): profile(FruitLite(variant="L2"), s) for s in (224, 32)}
    if not all(r["eligible"] for r in profiles.values()):
        raise ValueError("Architecture exceeds the exam budget")
    loaded = {}
    for size, config in configs.items():
        probe = dataset(root, size, train_rows)
        probe_loader = loader(probe, 2, config.workers, False, seed, False)
        images, targets = next(iter(probe_loader))
        with torch.no_grad():
            output = FruitLite(variant="L2").eval()(images)
        if output.shape != (2, 5) or not torch.isfinite(output).all():
            raise ValueError("Input/model probe failed")
        loaded[str(size)] = {
            "input_shape": list(images.shape),
            "output_shape": list(output.shape),
            "workers": config.workers,
        }
    preflight = {
        "status": "preflight_passed",
        "profile": profiles,
        "loader": loaded,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "torchvision", "Pillow", "numpy", "scikit-learn", "matplotlib")
        },
        "python": __import__("sys").executable,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if source_digest() != plan["source_sha256"]:
        raise ValueError("Source code changed during preflight; preserve this record and use a new --run-name")
    write_json(folder / "preflight.json", preflight)
    # Keep the exact implementation beside run settings for later reporting.
    for source in (Path(__file__).parent).glob("*.py"):
        destination = folder / "source/src" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for source in (root / "configs").glob("*.json"):
        destination = folder / "source/configs" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    if (root / "train.py").exists():
        shutil.copy2(root / "train.py", folder / "source/train.py")
    if check_only:
        print("PREFLIGHT PASSED. No training epochs or checkpoints created.", flush=True)
        return preflight
    final_configs = {}
    for size, config in configs.items():
        size_folder = folder / f"Mid{size}"
        development = train_job(
            config,
            size_folder / "development",
            dataset(root, size, train_rows, True),
            dataset(root, size, val_rows),
            {**provenance, "purpose": "development"},
        )
        plot_history(size_folder / "development")
        selected_epochs = development["best_epoch"]
        final_config = replace(config, epochs=selected_epochs)
        selection = {
            "epoch_budget": selected_epochs,
            "selected_weights": development["selected_weights"],
            "development_validation_accuracy": development["best_validation_accuracy"],
            "selection_rule": "validation_top1_then_loss",
            "test_used": False,
        }
        write_json(size_folder / "final_selection.json", selection)
        full_pool = sorted((r for r in rows if r["split"] == "train"), key=lambda r: r["id"])
        train_job(
            final_config,
            size_folder / "final",
            dataset(root, size, full_pool, True),
            None,
            {**provenance, "purpose": "final_full_pool_refit", "selection": selection},
            selection_kind=development["selected_weights"],
        )
        plot_history(size_folder / "final")
        final_configs[size] = final_config
    # Both training plans finish before any test prediction is made.
    results = {}
    for size, config in final_configs.items():
        test_rows = sorted((r for r in rows if r["split"] == "test"), key=lambda r: r["id"])
        measured = evaluate_locked(folder / f"Mid{size}/final", dataset(root, size, test_rows), config)
        results[str(size)] = {
            "test_accuracy": measured["metrics"]["accuracy"],
            "test_macro_f1": measured["metrics"]["macro_f1"],
            "epoch_budget": config.epochs,
            "metrics_path": f"Mid{size}/final/test/metrics.json",
        }
    result = {"status": "completed", "workflow_identity": plan_identity, "results": results}
    write_json(folder / "summary.json", result)
    return result
