"""Overfit one fixed pilot crop per class, without claiming benchmark accuracy."""

import argparse

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from .common import CLASSES, ROOT, read_csv, write_json
from .model import FruitLite
from .train import seed_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    seed_all(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_csv(ROOT / "data" / "metadata" / "downloads.csv")
    selected = []
    for fruit in CLASSES:
        candidates = [row for row in rows if row["class"] == fruit and row["status"] == "pending_review"]
        if not candidates:
            raise ValueError(f"No pilot for {fruit}")
        selected.append(sorted(candidates, key=lambda row: row["id"])[0])
    results = {}
    for size in (32, 224):
        images = []
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
        for row in selected:
            with Image.open(ROOT / row["crop_path"]) as opened:
                image = ImageOps.pad(opened.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS, color=(127, 127, 127))
                images.append(transform(image))
        x = torch.stack(images).to(device)
        y = torch.arange(5, device=device)
        model = FruitLite(dropout=0).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.002)
        losses = []
        for step in range(args.steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            logits = model(x)
            accuracy = (logits.argmax(1) == y).float().mean().item()
            evaluation_loss = torch.nn.functional.cross_entropy(logits, y).item()
        results[str(size)] = {"purpose": "UNREVIEWED_PILOT_OVERFIT_ONLY_NOT_VALIDATION_OR_TEST", "steps": args.steps,
                              "initial_loss": losses[0], "last_training_loss": losses[-1], "evaluation_loss_same_images": evaluation_loss,
                              "accuracy_same_five_images": accuracy, "loss_history": losses, "ids": [row["id"] for row in selected]}
        print(f"Pilot overfit {size}: {losses[0]:.4f} -> {losses[-1]:.4f}; same-image eval {accuracy:.1%}", flush=True)
    write_json(ROOT / "outputs" / "pilot_overfit.json", results)
    if any(result["last_training_loss"] >= result["initial_loss"] or result["accuracy_same_five_images"] < .99 for result in results.values()):
        raise SystemExit("Overfit smoke check failed; do not treat the training pipeline as validated")


if __name__ == "__main__":
    main()