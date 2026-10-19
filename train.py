"""Run this file to train FruitLite-L2 on Mid224 and Mid32, resume and evaluate."""

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys


PYTHON = Path(r"C:\Users\anhna\AppData\Local\Microsoft\WindowsApps\python3.13.exe")


def verify_python():
    if not PYTHON.is_file():
        raise RuntimeError(f"Required Microsoft Store interpreter not found: {PYTHON}")
    result = subprocess.run([str(PYTHON), "-c", "import sys,json; print(json.dumps([sys.executable,sys.base_prefix]))"],
                            check=True, capture_output=True, text=True)
    executable, base = json.loads(result.stdout)
    if (sys.version_info[:2] != (3, 13) or sys.prefix != sys.base_prefix
            or os.path.normcase(os.path.realpath(sys.executable)) != os.path.normcase(os.path.realpath(executable))
            or os.path.normcase(sys.base_prefix) != os.path.normcase(base)):
        raise RuntimeError(f"Run this file with the selected base interpreter: {PYTHON}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="fruitlite_l2_seed2026", help="Change name to start a separate experiment")
    parser.add_argument("--sizes", nargs="+", type=int, choices=(224, 32), default=[224, 32])
    parser.add_argument("--epochs", type=int, help="Override development epochs for both resolutions")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check", action="store_true", help="Audit, profile and check real loaders without training")
    parser.add_argument("--full-audit", action="store_true", help="Force a full decoded image audit instead of reusing the valid cache")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive")
    verify_python()
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    from src.training_pipeline import run_pipeline
    try:
        run_pipeline(run_name=args.run_name, sizes=tuple(args.sizes), epochs=args.epochs,
                     batch_size=args.batch_size, workers=args.workers, device=args.device, check_only=args.check,
                     full_audit=args.full_audit)
    except KeyboardInterrupt:
        print("\nStopped. Run train.py again to resume from the last completed epoch.", flush=True)
        return 130
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
