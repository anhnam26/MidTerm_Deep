"""Reuse a decoded audit only after checking every frozen image's encoded hash."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import PIL

from . import input_dataset as inputs
from .common import write_json


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validation_key(root, rows):
    return _digest(
        {
            "schema": 1,
            "root": str(Path(root).resolve()),
            "rows": rows,
            "classes": inputs.CLASSES,
            "quotas": inputs.SPLIT_QUOTAS,
            "limits": inputs.LIMITS,
            "formats": inputs.FORMATS,
            "pillow": PIL.__version__,
            "validator_sha256": hashlib.sha256(Path(inputs.__file__).read_bytes()).hexdigest(),
            "cache_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )


def _image_inventory(root):
    """Scan without following links or resolving each individual image path."""
    actual = {}
    base = root / "DataNew"
    if base.is_symlink() or base.is_junction():
        raise ValueError("Input inventory contains a linked dataset directory")
    for size in inputs.FORMATS:
        scope = base / f"Mid{size}"
        if not scope.is_dir():
            raise ValueError(f"Input inventory is missing Mid{size}")
        pending = [scope]
        while pending:
            directory = pending.pop()
            if directory.is_symlink() or directory.is_junction():
                raise ValueError(f"Input inventory contains a linked directory: {directory}")
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        raise ValueError(f"Input inventory contains a symbolic link: {entry.path}")
                    path = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                    elif entry.is_file(follow_symlinks=False):
                        actual[path.relative_to(root).as_posix()] = path
                    else:
                        raise ValueError(f"Unsupported input inventory entry: {path}")
    return actual


def verify_encoded_files(root, rows):
    root = Path(root).resolve()
    expected = {
        Path(row[f"mid{size}_path"]).as_posix(): row[f"mid{size}_sha256"] for row in rows for size in inputs.FORMATS
    }
    actual = _image_inventory(root)
    if len(expected) != len(rows) * len(inputs.FORMATS) or actual.keys() != expected.keys():
        raise ValueError("Input inventory changed: missing, duplicate or unmanifested files")

    def check(item):
        relative, path = item
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[relative]:
            raise ValueError(f"Input image hash changed after freeze: {relative}")

    items = list(actual.items())
    last_progress = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        # Bound pending tasks instead of creating 50,500 futures at once.
        for start in range(0, len(items), 512):
            for _ in pool.map(check, items[start : start + 512]):
                pass
            if time.monotonic() - last_progress >= 10:
                print(f"INPUT_CHECK: {min(start + 512, len(items))}/{len(items)} hashes", flush=True)
                last_progress = time.monotonic()


def validate_inputs(root, rows, cache_path, *, force_full=False):
    """Return (full audit report, reused); never accept mtime/size alone."""
    key = validation_key(root, rows)
    cache_path = Path(cache_path)
    cached = None
    if not force_full and cache_path.is_file():
        try:
            candidate = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                isinstance(candidate, dict)
                and candidate.get("schema") == 1
                and candidate.get("key") == key
                and isinstance(candidate.get("report"), dict)
                and candidate["report"].get("structural_pass") is True
                and candidate.get("report_sha256") == _digest(candidate["report"])
            ):
                cached = candidate["report"]
        except (OSError, ValueError, TypeError):
            pass
    if cached is not None:
        verify_encoded_files(root, rows)
        return cached, True
    report = inputs.validate(root, rows)
    verify_encoded_files(root, rows)
    if validation_key(root, rows) != key:
        raise ValueError("Input validation policy/code changed during audit")
    write_json(cache_path, {"schema": 1, "key": key, "report_sha256": _digest(report), "report": report})
    return report, False
