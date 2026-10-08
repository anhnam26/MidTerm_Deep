"""GPU step benchmark on random tensors: memory/timing only, NOT accuracy."""

import argparse
import time

import torch

from .common import ROOT, write_json
from .model import FruitLite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, choices=[32, 224], default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if args.batch_size < 2 or args.steps < 1:
        parser.error("batch-size >=2, steps >=1 required")
    if not torch.cuda.is_available():
        parser.error("CUDA benchmark requires an available GPU")
    model = FruitLite().cuda().train()
    images = torch.randn(args.batch_size, 3, args.size, args.size, device="cuda")
    labels = torch.arange(args.batch_size, device="cuda") % 5
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
    scaler = torch.amp.GradScaler("cuda")

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda"):
            loss = torch.nn.functional.cross_entropy(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    result = {"purpose": "RANDOM_TENSOR_TRAIN_STEP_ONLY_NO_IO_OR_AUGMENTATION",
              "gpu": torch.cuda.get_device_name(0), "size": args.size, "batch_size": args.batch_size,
              "steps": args.steps, "seconds_per_step": seconds / args.steps,
              "images_per_second": args.batch_size * args.steps / seconds,
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2 ** 20}
    write_json(ROOT / "outputs" / f"benchmark{args.size}_batch{args.batch_size}.json", result)
    print(result)


if __name__ == "__main__":
    main()