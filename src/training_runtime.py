"""Completed-epoch transactions, live logs, metrics and resumable FruitLite training."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch import nn

from .common import CLASSES, write_csv, write_json
from .model import FruitLite
from .profile import profile
from .train import learning_rate, loader, seed_all


@dataclass(frozen=True)
class TrainConfig:
    size: int = 224
    epochs: int = 180
    batch_size: int = 64
    workers: int = 4
    lr: float = .0015
    weight_decay: float = .01
    warmup: int = 5
    label_smoothing: float = .05
    dropout: float = .2
    mixup_alpha: float = .2
    mixup_probability: float = .25
    ema_decay: float = .999
    grad_clip: float = 5.
    seed: int = 2026
    device: str = "auto"
    log_interval: int = 20

    def validate(self):
        if (self.size not in (32, 224) or self.epochs < 1 or self.batch_size < 2
                or self.workers < 0 or self.lr <= 0 or self.weight_decay < 0
                or self.warmup < 0 or not 0 <= self.label_smoothing < 1
                or not 0 <= self.dropout < 1 or self.mixup_alpha <= 0
                or not 0 <= self.mixup_probability <= 1 or not 0 <= self.ema_decay < 1
                or self.grad_clip <= 0 or self.log_interval < 1
                or self.device not in ("auto", "cpu", "cuda")):
            raise ValueError("Invalid training configuration")
        if not all(math.isfinite(v) for v in asdict(self).values() if isinstance(v, float)):
            raise ValueError("Training settings must be finite")


def source_digest():
    digest = hashlib.sha256()
    for name in ("model.py", "profile.py", "common.py", "train.py", "training_data.py",
                 "input_dataset.py", "input_audit_cache.py", "training_runtime.py", "training_pipeline.py"):
        path = Path(__file__).with_name(name)
        if path.exists():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    entrypoint = Path(__file__).parents[1] / "train.py"
    if entrypoint.exists():
        digest.update(b"root/train.py")
        digest.update(entrypoint.read_bytes())
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_run(path):
    """OS-owned lock releases even on process death; the marker may remain safely."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another process owns this training run: {path}") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


class RunLog:
    def __init__(self, folder):
        self.console = (folder / "console.log").open("a", encoding="utf-8", buffering=1)
        self.batches = (folder / "batches.jsonl").open("a", encoding="utf-8", buffering=1)
        self.attempt = datetime.now(timezone.utc).isoformat()

    def say(self, message, *, terminal=True):
        line = f"[{datetime.now().astimezone().strftime('%H:%M:%S')}] {message}"
        if terminal:
            print(line, flush=True)
        self.console.write(line + "\n")

    def batch(self, entry):
        self.batches.write(json.dumps({"attempt": self.attempt, **entry}, allow_nan=False) + "\n")

    def close(self):
        self.console.close()
        self.batches.close()


def rng_state():
    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy[0], numpy[1].tolist(), *numpy[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def update_ema(ema, model, decay, updates):
    decay = min(decay, (1 + updates) / (10 + updates))
    current = model.state_dict()
    for name, value in ema.state_dict().items():
        # Running BN moments are copied: their own exponential averaging is enough.
        if value.is_floating_point() and "running_" not in name:
            value.lerp_(current[name], 1 - decay)
        else:
            value.copy_(current[name])


def classification_metrics(labels, probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)
    predictions = probabilities.argmax(1)
    report = classification_report(labels, predictions, labels=list(range(5)),
                                   target_names=CLASSES, output_dict=True, zero_division=0)
    confidence = probabilities.max(1)
    correct = predictions == labels
    ece = 0.
    for low, high in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        selected = (confidence > low) & (confidence <= high)
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    auroc = None
    if len(np.unique(labels)) == 5:
        # Normalize float32 softmax sums for sklearn's probability tolerance.
        normalized = probabilities / probabilities.sum(1, keepdims=True)
        auroc = float(roc_auc_score(labels, normalized, labels=list(range(5)), multi_class="ovr", average="macro"))
    return {"accuracy": float(correct.mean()), "correct": int(correct.sum()), "total": len(labels),
            "macro_precision": report["macro avg"]["precision"], "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"], "weighted_f1": report["weighted avg"]["f1-score"],
            "macro_auroc_ovr": auroc, "ece_15bins": float(ece),
            "brier": float(((probabilities - np.eye(5)[labels]) ** 2).sum(1).mean()),
            "per_class": {name: report[name] for name in CLASSES},
            "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(5))).tolist()}


def run_epoch(model, data_loader, device, config, log, *, epoch, phase,
              optimizer=None, scaler=None, ema=None, ema_updates=0):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = samples = skipped = mixed_batches = 0
    grad_sum = grad_max = 0.
    finite_gradients = 0
    labels, probabilities = [], []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing if training else 0.)
    with torch.set_grad_enabled(training):
        for batch_index, (images, targets) in enumerate(data_loader, 1):
            step_start = time.perf_counter()
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            partner, weight = targets, 1.
            if training and random.random() < config.mixup_probability:
                weight = float(np.random.beta(config.mixup_alpha, config.mixup_alpha))
                permutation = torch.randperm(len(targets), device=device)
                images = images * weight + images[permutation] * (1 - weight)
                partner = targets[permutation]
                mixed_batches += 1
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = weight * criterion(logits, targets) + (1 - weight) * criterion(logits, partner)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch} {phase} batch {batch_index}")
            grad_norm = None
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                previous_scale = scaler.get_scale()
                grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=False))
                if not math.isfinite(grad_norm):
                    if not scaler.is_enabled():
                        raise FloatingPointError("Nonfinite gradients in full precision")
                    # AMP can overflow at its initial scale despite finite loss.
                    # Skip this update and reduce scale; never corrupt model/EMA.
                    scaler.update(new_scale=previous_scale / 2)
                    grad_norm = None
                    skipped += 1
                    log.say(f"AMP gradient overflow: skipped update; scale {previous_scale:g} -> {scaler.get_scale():g}", terminal=False)
                else:
                    grad_sum += grad_norm
                    grad_max = max(grad_max, grad_norm)
                    finite_gradients += 1
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= previous_scale:
                        ema_updates += 1
                        update_ema(ema, model, config.ema_decay, ema_updates)
                    else:
                        skipped += 1
            scores = logits.detach().float().softmax(1)
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Nonfinite model probabilities")
            batch_correct = int((scores.argmax(1) == targets).sum())
            loss_sum += float(loss.detach()) * len(targets)
            correct += batch_correct
            samples += len(targets)
            labels.extend(targets.cpu().tolist())
            probabilities.extend(scores.cpu().tolist())
            seconds = time.perf_counter() - started
            log.batch({"epoch": epoch, "phase": phase, "batch": batch_index,
                       "batches": len(data_loader), "samples_seen": samples, "batch_loss": float(loss.detach()),
                       "running_loss": loss_sum / samples, "running_accuracy": correct / samples,
                       "lr": optimizer.param_groups[0]["lr"] if training else None,
                       "gradient_norm_before_clip": grad_norm, "mixup_weight": weight,
                       "amp_scale": scaler.get_scale() if training else None,
                       "step_seconds": time.perf_counter() - step_start, "elapsed_seconds": seconds})
            if batch_index == 1 or batch_index % config.log_interval == 0 or batch_index == len(data_loader):
                eta = seconds / batch_index * (len(data_loader) - batch_index)
                log.say(f"{phase:7s} epoch={epoch} batch={batch_index}/{len(data_loader)} "
                        f"loss={loss_sum / samples:.4f} acc={correct / samples:.2%} "
                        f"speed={samples / seconds:.1f} img/s ETA={eta / 60:.1f}m", terminal=False)
    if not samples:
        raise ValueError("Empty data loader")
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    metrics = classification_metrics(labels, probabilities)
    metrics.update({"loss": loss_sum / samples, "seconds": seconds, "images_per_second": samples / seconds,
                    "gradient_norm_mean": grad_sum / max(1, finite_gradients) if training else None,
                    "gradient_norm_max": grad_max if training else None, "amp_skipped_steps": skipped,
                    "mixed_batches": mixed_batches, "accuracy_context": "augmented_primary_labels" if training else "unaugmented",
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else 0})
    log.say(f"{phase} completed loss={metrics['loss']:.4f} top1={metrics['accuracy']:.2%} "
            f"precision={metrics['macro_precision']:.4f} recall={metrics['macro_recall']:.4f} "
            f"F1={metrics['macro_f1']:.4f} GPU_peak={metrics['peak_allocated_mib']:.0f} MiB", terminal=False)
    if not training:
        for fruit, item in metrics["per_class"].items():
            log.say(f"  {fruit:10s} P={item['precision']:.4f} R={item['recall']:.4f} "
                    f"F1={item['f1-score']:.4f} samples={item['support']:.0f}", terminal=False)
    return metrics, {"labels": labels, "probabilities": probabilities}, ema_updates


def export_epoch_files(folder, checkpoint):
    history = checkpoint["history"]
    write_json(folder / "epochs.json", history)
    flat = []
    for entry in history:
        row = {key: entry[key] for key in ("epoch", "lr", "epoch_seconds", "best_epoch", "best_accuracy", "selected_weights")}
        for phase in ("train", "val", "ema_val"):
            if entry[phase]:
                row.update({f"{phase}_{key}": value for key, value in entry[phase].items()
                            if not isinstance(value, (dict, list))})
        flat.append(row)
    write_csv(folder / "epochs.csv", flat, fields=list(dict.fromkeys(k for row in flat for k in row)))
    best = {"model": checkpoint["best_weights"], "model_settings": checkpoint["model_settings"],
            "epoch": checkpoint["best_epoch"], "size": checkpoint["config"]["size"],
            "selected_weights": checkpoint["best_kind"], "validation": checkpoint["best_validation"],
            "identity": checkpoint["identity"], "selection_rule": checkpoint["selection_rule"],
            "provenance": checkpoint["provenance"]}
    atomic_save(best, folder / "best.pt")
    summary = {"status": "completed" if checkpoint["epoch"] == checkpoint["config"]["epochs"] else "running",
               "completed_epochs": checkpoint["epoch"], "best_epoch": checkpoint["best_epoch"],
               "best_validation_accuracy": checkpoint["best_validation"]["accuracy"] if checkpoint["best_validation"] else None,
               "selected_weights": checkpoint["best_kind"], "selection_rule": checkpoint["selection_rule"],
               "test_used_during_training": False}
    write_json(folder / "summary.json", summary)
    return summary


def train_job(config, folder, train_data, validation, provenance, selection_kind="ema"):
    config.validate()
    if selection_kind not in ("raw", "ema"):
        raise ValueError("Unknown final selection weights")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with exclusive_run(folder / ".train.lock"):
        return _train_job(config, folder, train_data, validation, provenance, selection_kind)


def _train_job(config, folder, train_data, validation, provenance, selection_kind):
    if not (folder / 'last.pt').exists() and any((folder / name).exists() for name in
            ('best.pt', 'epochs.csv', 'epochs.json', 'summary.json')):
        raise ValueError('Missing last.pt in an existing training run; restore the checkpoint instead of restarting')
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else
                          "cpu" if config.device == "auto" else config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    settings = {"config": asdict(config), "model_settings": {"variant": "L2", "dropout": config.dropout},
                "optimizer": "AdamW", "betas": [.9, .999], "decay_policy": "weights_only_no_bias_or_BN",
                "schedule": "linear warmup + cosine to 0.01*lr", "source_sha256": source_digest(),
                "provenance": provenance, "train_images": len(train_data),
                "validation_images": len(validation) if validation is not None else 0,
                "selection_kind": selection_kind, "torch": str(torch.__version__), "device": str(device),
                "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                "validation_persistent_workers": config.workers > 0 and validation is not None,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "cudnn_benchmark": False, "cudnn_deterministic": True}
    identity = fingerprint(settings)
    settings_path = folder / "settings.json"
    if settings_path.exists():
        previous = json.loads(settings_path.read_text(encoding="utf-8"))
        if previous["identity"] != identity:
            raise ValueError("Run settings, source code or data changed; use --run-name with a new name")
    elif (folder / "last.pt").exists() or (folder / "best.pt").exists():
        raise ValueError("Checkpoint without settings; preserve the run and use a new run name")
    seed_all(config.seed)
    model = FruitLite(**settings["model_settings"]).to(device)
    properties = {str(size): profile(model, size) for size in (224, 32)}
    if not all(item["eligible"] for item in properties.values()):
        raise ValueError("Model violates the exam parameter/FLOP limits")
    ema = deepcopy(model).eval().requires_grad_(False)
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": config.weight_decay},
                                  {"params": no_decay, "weight_decay": 0.}], lr=config.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    state = None
    if (folder / "last.pt").exists():
        state = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
        if state["identity"] != identity:
            raise ValueError("Checkpoint identity changed")
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        restore_rng(state["rng"])
        summary = export_epoch_files(folder, state)  # repair exports after interrupted writes
        if state["epoch"] == config.epochs:
            print(f"SKIP {folder.name}: completed {config.epochs} epochs", flush=True)
            return summary
    else:
        write_json(settings_path, {"identity": identity, **settings, "profile": properties,
                                  "model": str(model), "classes": list(CLASSES),
                                  "python": os.sys.version, "python_executable": os.sys.executable,
                                  "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                                  "cuda_runtime": torch.version.cuda,
                                  "cudnn": torch.backends.cudnn.version(), "test_used_during_training": False})
    training_loader = loader(train_data, config.batch_size, config.workers, True, config.seed, device.type == "cuda")
    val_loader = loader(validation, config.batch_size, config.workers, False, config.seed, device.type == "cuda", persistent=True) if validation is not None else None
    log = RunLog(folder)
    start_epoch = state["epoch"] if state else 0
    history = state["history"] if state else []
    best_score = tuple(state["best_score"]) if state else (-1., -float("inf"))
    best_weights = state["best_weights"] if state else None
    best_epoch = state["best_epoch"] if state else 0
    best_kind = state["best_kind"] if state else selection_kind
    best_validation = state["best_validation"] if state else None
    updates = state["ema_updates"] if state else 0
    committed_epoch = start_epoch
    if state is not None:
        log.say(f"RESUME Mid{config.size}/{provenance.get('purpose', folder.name)} from {folder / 'last.pt'}; "
                f"epoch={start_epoch} -> next_epoch={start_epoch + 1}; "
                "model/optimizer/AMP/EMA/RNG/history restored")
    else:
        log.say(f"START Mid{config.size} device={device} train={len(train_data)} val={len(validation) if validation is not None else 0} "
                f"epochs={config.epochs} no checkpoint yet; first epoch=1 output={folder}")
    log.say(f"FruitLite-L2 params={properties[str(config.size)]['learnable_parameters']:,} "
            f"FLOPs={properties[str(config.size)]['flops']:,}; best selection={'validation top1, then loss' if validation is not None else 'fixed final epoch'}", terminal=False)
    log.say("CONFIG " + json.dumps(asdict(config), sort_keys=True), terminal=False)
    try:
        for epoch_index in range(start_epoch, config.epochs):
            # Fresh workers and epoch-specific RNG make interrupted epochs replayable.
            seed_all(config.seed + epoch_index)
            training_loader.generator.manual_seed(config.seed + epoch_index)
            epoch_started = time.perf_counter()
            lr = learning_rate(epoch_index, config.epochs, config.warmup, config.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            train_metrics, _, updates = run_epoch(model, training_loader, device, config, log,
                epoch=epoch_index + 1, phase="train", optimizer=optimizer, scaler=scaler, ema=ema, ema_updates=updates)
            raw_val = ema_val = None
            if val_loader is not None:
                raw_val = run_epoch(model, val_loader, device, config, log, epoch=epoch_index + 1, phase="val")[0]
                ema_val = run_epoch(ema, val_loader, device, config, log, epoch=epoch_index + 1, phase="ema_val")[0]
                candidates = [(raw_val, model, "raw"), (ema_val, ema, "ema")]
                candidate, selected, kind = max(candidates, key=lambda x: (x[0]["accuracy"], -x[0]["loss"]))
                score = (candidate["accuracy"], -candidate["loss"])
            else:
                candidate, selected, kind = None, ema if selection_kind == "ema" else model, selection_kind
                score = (float(epoch_index + 1), 0.)
            if score > best_score:
                best_score, best_weights, best_epoch = score, cpu_state(selected), epoch_index + 1
                best_kind, best_validation = kind, candidate
            entry = {"epoch": epoch_index + 1, "lr": lr, "train": train_metrics, "val": raw_val, "ema_val": ema_val,
                     "epoch_seconds": time.perf_counter() - epoch_started, "best_epoch": best_epoch,
                     "best_accuracy": best_validation["accuracy"] if best_validation else None, "selected_weights": best_kind}
            history.append(entry)
            checkpoint = {"schema": 1, "identity": identity, "config": asdict(config), "provenance": provenance,
                "model_settings": settings["model_settings"], "epoch": epoch_index + 1, "model": cpu_state(model),
                "ema": cpu_state(ema), "ema_updates": updates, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "rng": rng_state(), "history": history, "best_score": best_score, "best_weights": best_weights,
                "best_epoch": best_epoch, "best_kind": best_kind, "best_validation": best_validation,
                "selection_rule": "validation_top1_then_loss" if validation is not None else "fixed_final_epoch"}
            # last.pt is the commit point. best/CSV/JSON can be rebuilt entirely from it.
            atomic_save(checkpoint, folder / "last.pt")
            committed_epoch = epoch_index + 1
            summary = export_epoch_files(folder, checkpoint)
            val_text = f" val_loss={raw_val['loss']:.4f} val={raw_val['accuracy']:.2%} ema_val={ema_val['accuracy']:.2%} best={best_validation['accuracy']:.2%}@{best_epoch}" if raw_val else " fixed-final-epoch selection"
            log.say(f"Mid{config.size}/{provenance.get('purpose', folder.name)} EPOCH {epoch_index + 1}/{config.epochs} lr={lr:.7f} train_loss={train_metrics['loss']:.4f} "
                    f"train_acc={train_metrics['accuracy']:.2%}{val_text} seconds={entry['epoch_seconds']:.1f}; last.pt/best.pt saved")
        return summary
    except BaseException as exc:
        log.say(f"STOP {type(exc).__name__}: {exc}; resume from the last completed epoch")
        log.console.write(traceback.format_exc())
        write_json(folder / "interruption.json", {"time": log.attempt, "exception": type(exc).__name__,
                   "message": str(exc), "completed_epochs": committed_epoch, "partial_epoch_checkpoint_saved": False})
        raise
    finally:
        log.close()
