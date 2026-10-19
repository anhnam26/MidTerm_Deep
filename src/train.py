"""Train/evaluate FruitLite-L. Test is never evaluated by the training command."""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .common import CLASSES, ROOT, read_csv, write_json
from .model import FruitLite
from .profile import profile


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def image_transform(size, training):
    operations = []
    if training:
        operations.extend([
            transforms.RandomResizedCrop(size, scale=(0.85 if size == 32 else 0.75, 1.0), ratio=(0.9, 1.1), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10, fill=(127, 127, 127)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.015),
        ])
    operations.append(transforms.ToTensor())
    # Standard initial normalization, explicitly recorded; never estimated on test.
    operations.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    return transforms.Compose(operations)


class ManifestDataset(Dataset):
    def __init__(self, size, split, training=False, rows=None, root=None):
        self.root = Path(root) if root is not None else ROOT
        self.rows = list(rows) if rows is not None else [row for row in read_csv(self.root / "data" / "metadata" / "splits.csv")
                     if row["split"] == split or (split == "train_pool" and row["split"] in ("train", "val"))]
        if not self.rows:
            raise ValueError(f"No rows for split {split}")
        self.size = size
        self.transform = image_transform(size, training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if row.get(f"mid{self.size}_path"):
            from .input_dataset import image_path
            path = image_path(self.root, row, self.size)
        else:
            path = self.root / "data" / f"Mid{self.size}" / row["split"] / row["class"] / f"{row['id']}.png"
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (self.size, self.size):
                raise ValueError(f"Wrong dataset image: {path}")
            tensor = self.transform(image)
        return tensor, CLASSES.index(row["class"])


def loader(dataset, batch_size, workers, training, seed, cuda, persistent=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=training, num_workers=workers,
                      pin_memory=cuda, worker_init_fn=seed_worker, generator=generator,
                      persistent_workers=bool(persistent and workers > 0))


def run_epoch(model, data_loader, device, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = total = 0
    labels, predictions, confidences = [], [], []
    started = time.perf_counter()
    with torch.set_grad_enabled(training):
        for images, targets in data_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(images)
                loss = criterion(output, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            confidence, predicted = output.detach().float().softmax(dim=1).max(dim=1)
            total_loss += loss.item() * targets.numel()
            correct += (predicted == targets).sum().item()
            total += targets.numel()
            labels.extend(targets.cpu().tolist())
            predictions.extend(predicted.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
    if device.type == "cuda":
        torch.cuda.synchronize()
    if total == 0:
        raise ValueError("Empty data loader")
    return {"loss": total_loss / total, "accuracy": correct / total, "correct": correct,
            "total": total, "seconds": time.perf_counter() - started}, labels, predictions, confidences


def learning_rate(epoch, epochs, warmup, base):
    if epoch < warmup:
        return base * (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, epochs - warmup - 1)
    return base * (0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2)


def split_digest():
    import hashlib
    return hashlib.sha256((ROOT / "data" / "metadata" / "splits.csv").read_bytes()).hexdigest()


def train(args):
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}. Use a new run name.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        if args.final:
            raise ValueError("Smoke data cannot be used for a final run")
        train_data = datasets.FakeData(size=20, image_size=(3, args.size, args.size), num_classes=5, transform=image_transform(args.size, True), random_offset=0)
        validation = datasets.FakeData(size=10, image_size=(3, args.size, args.size), num_classes=5, transform=image_transform(args.size, False), random_offset=1000)
        digest = "FAKE_DATA_SMOKE_ONLY"
    else:
        from .data import audit
        audit()  # refuses incomplete/malformed official datasets
        train_data = ManifestDataset(args.size, "train_pool" if args.final else "train", True)
        validation = None if args.final else ManifestDataset(args.size, "val", False)
        digest = split_digest()
    model_settings = {"multiscale": not args.no_multiscale, "attention": not args.no_attention, "dropout": args.dropout}
    model = FruitLite(**model_settings).to(device)
    properties = {str(size): profile(model, size) for size in (224, 32)}
    if not all(item["eligible"] for item in properties.values()):
        raise ValueError("Model exceeds parameter/FLOPs limits")
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    train_loader = loader(train_data, args.batch_size, args.workers, True, args.seed, device.type == "cuda")
    val_loader = loader(validation, args.batch_size, args.workers, False, args.seed, device.type == "cuda") if validation is not None else None
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update({"device": str(device), "torch_version": str(torch.__version__), "classes": CLASSES,
                     "model_settings": model_settings, "profile": properties, "split_sha256": digest,
                     "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                     "purpose": "SMOKE_ONLY_NOT_MIDTERM_RESULTS" if args.smoke else "final" if args.final else "development",
                     "train_images": len(train_data), "val_images": len(validation) if validation is not None else 0,
                     "test_used_during_training": False})
    write_json(output_dir / "settings.json", settings)
    best_accuracy = -1
    best_epoch = 0
    log_fields = ["epoch", "lr", "train_loss", "train_accuracy", "train_seconds", "val_loss", "val_accuracy", "val_seconds"]
    with (output_dir / "epochs.csv").open("w", newline="", encoding="utf-8") as log:
        writer = csv.DictWriter(log, fieldnames=log_fields)
        writer.writeheader()
        for epoch in range(args.epochs):
            lr = learning_rate(epoch, args.epochs, args.warmup, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            train_metrics, _, _, _ = run_epoch(model, train_loader, device, criterion, optimizer, scaler)
            val_metrics = run_epoch(model, val_loader, device, criterion)[0] if val_loader is not None else None
            entry = {"epoch": epoch + 1, "lr": lr, "train_loss": train_metrics["loss"],
                     "train_accuracy": train_metrics["accuracy"], "train_seconds": train_metrics["seconds"],
                     "val_loss": val_metrics["loss"] if val_metrics else "", "val_accuracy": val_metrics["accuracy"] if val_metrics else "",
                     "val_seconds": val_metrics["seconds"] if val_metrics else ""}
            writer.writerow(entry)
            log.flush()
            checkpoint = {"model": model.state_dict(), "model_settings": model_settings, "size": args.size,
                          "epoch": epoch + 1, "settings": settings, "optimizer": optimizer.state_dict(),
                          "scaler": scaler.state_dict(), "validation": val_metrics}
            torch.save(checkpoint, output_dir / "last.pt")
            if val_metrics and val_metrics["accuracy"] > best_accuracy:
                best_accuracy, best_epoch = val_metrics["accuracy"], epoch + 1
                torch.save(checkpoint, output_dir / "best.pt")
            print(json.dumps(entry), flush=True)
    write_json(output_dir / "summary.json", {"purpose": settings["purpose"], "completed_epochs": args.epochs,
                                             "best_validation_accuracy": best_accuracy if val_loader else None,
                                             "best_epoch": best_epoch if val_loader else None,
                                             "final_checkpoint": "last.pt" if args.final else "best.pt"})


def evaluate(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Evaluation output already exists; preserve previous results")
    # Only load trusted checkpoints. weights_only=True prevents arbitrary pickle code.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["settings"]["purpose"] == "SMOKE_ONLY_NOT_MIDTERM_RESULTS":
        raise ValueError("Cannot use a FakeData smoke checkpoint for official evaluation")
    if args.split == "test" and checkpoint["settings"]["purpose"] != "final":
        raise ValueError("Test requires a locked final run; select hyperparameters using validation")
    if split_digest() != checkpoint["settings"]["split_sha256"]:
        raise ValueError("Dataset split changed since training")
    from .data import audit
    audit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FruitLite(**checkpoint["model_settings"]).to(device)
    model.load_state_dict(checkpoint["model"])
    dataset = ManifestDataset(checkpoint["size"], args.split, False)
    data_loader = loader(dataset, args.batch_size, args.workers, False, 2026, device.type == "cuda")
    metrics, labels, predictions, confidence = run_epoch(model, data_loader, device, nn.CrossEntropyLoss())
    report = classification_report(labels, predictions, labels=list(range(5)), target_names=CLASSES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(labels, predictions, labels=list(range(5)))
    write_json(args.output / "metrics.json", {"split": args.split, "checkpoint": str(args.checkpoint.resolve()),
                                             "epoch": checkpoint["epoch"], "metrics": metrics, "classification_report": report,
                                             "confusion_matrix": matrix.tolist(), "classes": CLASSES})
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "true_class", "predicted_class", "confidence"])
        for row, label, prediction, score in zip(dataset.rows, labels, predictions, confidence):
            writer.writerow([row["id"], CLASSES[label], CLASSES[prediction], score])
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--config", type=Path)
    tr.add_argument("--size", type=int, choices=[224, 32], default=224)
    tr.add_argument("--epochs", type=int, default=150)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--lr", type=float, default=0.05)
    tr.add_argument("--momentum", type=float, default=0.9)
    tr.add_argument("--weight-decay", type=float, default=5e-4)
    tr.add_argument("--warmup", type=int, default=5)
    tr.add_argument("--label-smoothing", type=float, default=0.05)
    tr.add_argument("--dropout", type=float, default=0.2)
    tr.add_argument("--workers", type=int, default=0)
    tr.add_argument("--seed", type=int, default=2026)
    tr.add_argument("--output", type=Path, default=ROOT / "outputs" / "dev224_seed2026")
    tr.add_argument("--no-multiscale", action="store_true")
    tr.add_argument("--no-attention", action="store_true")
    tr.add_argument("--smoke", action="store_true", help="Explicit synthetic infrastructure test; NOT official results")
    tr.add_argument("--final", action="store_true", help="Train all 25000 train+val images using previously chosen fixed epochs")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--split", choices=["val", "test"], default="val")
    ev.add_argument("--batch-size", type=int, default=64)
    ev.add_argument("--workers", type=int, default=0)
    ev.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        if args.config:
            config = json.loads(args.config.read_text(encoding="utf-8"))
            valid = {action.dest for action in tr._actions} - {"help", "config", "command"}
            if set(config) - valid:
                parser.error(f"Unknown configuration keys: {set(config) - valid}")
            # Configuration overrides defaults; document this to avoid ambiguous runs.
            for key, value in config.items():
                setattr(args, key, Path(value) if key == "output" else value)
        if args.epochs < 1 or args.batch_size < 2 or args.workers < 0 or args.lr <= 0 or not 0 < args.momentum < 1 or args.warmup < 0 or args.size not in (32, 224) or not 0 <= args.label_smoothing < 1 or not 0 <= args.dropout < 1:
            parser.error("Invalid training settings")
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
