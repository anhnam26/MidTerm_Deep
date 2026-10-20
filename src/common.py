import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("apple", "banana", "orange", "strawberry", "pineapple")
CLASS_IDS = {
    "/m/014j1m": "apple",
    "/m/09qck": "banana",
    "/m/0cyhj_": "orange",
    "/m/07fbm7": "strawberry",
    "/m/0fp6w": "pineapple",
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    if not fields:
        raise ValueError("CSV needs explicit fields when empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
