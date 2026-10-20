"""Verify saved test results or classify one image with a completed checkpoint."""

import argparse
import json
from pathlib import Path

from train import verify_python


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="fruitlite_l2_seed2026")
    parser.add_argument("--image", type=Path, help="Optional image to classify")
    parser.add_argument("--size", type=int, choices=(224, 32), default=224)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args(argv)
    verify_python()
    import torch

    torch.set_num_threads(4)
    from src.evaluation import predict_image, print_results, read_results, run_folder

    folder = run_folder(args.run_name)
    if args.image:
        print(json.dumps(predict_image(folder, args.image, args.size, args.device), indent=2))
    else:
        print_results(read_results(folder))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
