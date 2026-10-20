"""Dataset, deterministic loaders and learning-rate helpers for FruitLite-L2."""

import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .common import CLASSES, ROOT, read_csv


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def image_transform(size, training):
    operations = []
    if training:
        operations.extend(
            [
                transforms.RandomResizedCrop(
                    size, scale=(0.85 if size == 32 else 0.75, 1.0), ratio=(0.9, 1.1), antialias=True
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10, fill=(127, 127, 127)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.015),
            ]
        )
    operations.append(transforms.ToTensor())
    # Standard initial normalization, explicitly recorded; never estimated on test.
    operations.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    return transforms.Compose(operations)


class ManifestDataset(Dataset):
    def __init__(self, size, split, training=False, rows=None, root=None):
        self.root = Path(root) if root is not None else ROOT
        self.rows = (
            list(rows)
            if rows is not None
            else [
                row
                for row in read_csv(self.root / "data" / "metadata" / "splits.csv")
                if row["split"] == split or (split == "train_pool" and row["split"] in ("train", "val"))
            ]
        )
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
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=cuda,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=bool(persistent and workers > 0),
    )


def learning_rate(epoch, epochs, warmup, base):
    if epoch < warmup:
        return base * (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, epochs - warmup - 1)
    return base * (0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2)
