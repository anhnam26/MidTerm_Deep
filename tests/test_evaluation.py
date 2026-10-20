import hashlib
import json

import numpy as np
from PIL import Image
import pytest
import torch

from src.common import CLASSES, read_csv, write_csv, write_json
from src import evaluation
from src.model import FruitLite
from src.training_runtime import classification_metrics, fingerprint


@pytest.fixture
def completed_run(tmp_path):
    folder = tmp_path / "outputs/fixture"
    final = folder / "Mid32/final"
    final.mkdir(parents=True)
    rows = [{"id": f"test_{i}", "class": c, "split": "test"} for i, c in enumerate(CLASSES)]
    write_csv(folder / "dataset_manifest.csv", rows)
    digest = hashlib.sha256((folder / "dataset_manifest.csv").read_bytes()).hexdigest()
    scores = np.full((5, 5), 0.05)
    np.fill_diagonal(scores, 0.8)
    model = FruitLite(variant="L2")
    best = {
        "model": model.state_dict(),
        "model_settings": {"variant": "L2"},
        "epoch": 2,
        "size": 32,
        "selected_weights": "ema",
        "identity": "fixture",
        "selection_rule": "fixed_final_epoch",
        "provenance": {"dataset_manifest_sha256": digest},
    }
    torch.save(best, final / "best.pt")
    torch.save({"epoch": 2, "identity": "fixture", "config": {"epochs": 2, "size": 32}}, final / "last.pt")
    measured = {
        "evaluation_identity": fingerprint(
            {
                "run_identity": "fixture",
                "weights_sha256": evaluation.weights_digest(best["model"]),
                "epoch": 2,
                "test_ids": [r["id"] for r in rows],
            }
        ),
        "checkpoint_sha256": hashlib.sha256((final / "best.pt").read_bytes()).hexdigest(),
        "epoch": 2,
        "selected_weights": "ema",
        "classes": list(CLASSES),
        "metrics": classification_metrics(list(range(5)), scores),
        "selection_used_test": False,
        "provenance": best["provenance"],
    }
    write_json(final / "test/metrics.json", measured)
    write_csv(
        final / "test/predictions.csv",
        [
            {
                "id": row["id"],
                "true_class": row["class"],
                "predicted_class": row["class"],
                "confidence": 0.8,
                **{f"p_{c}": s for c, s in zip(CLASSES, score)},
            }
            for row, score in zip(rows, scores)
        ],
    )
    write_json(
        folder / "summary.json",
        {"status": "completed", "results": {"32": {"epoch_budget": 2, "test_accuracy": 1.0, "test_macro_f1": 1.0}}},
    )
    return folder


def test_saved_results_are_verified_without_image_reads_or_file_changes(completed_run, monkeypatch):
    before = {p: p.read_bytes() for p in completed_run.rglob("*") if p.is_file()}
    monkeypatch.setattr(Image, "open", lambda *a, **kw: pytest.fail("No test inference expected"))
    assert evaluation.read_results(completed_run, (32,))["32"]["metrics"]["correct"] == 5
    assert before == {p: p.read_bytes() for p in completed_run.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "change", ["checkpoint", "incomplete", "probability", "label", "metric", "summary", "manifest"]
)
def test_corrupt_saved_results_are_rejected(completed_run, change):
    final = completed_run / "Mid32/final"
    if change == "checkpoint":
        with (final / "best.pt").open("ab") as f:
            f.write(b"changed")
    elif change == "incomplete":
        path = final / "last.pt"
        state = torch.load(path, weights_only=True)
        state["epoch"] = 1
        torch.save(state, path)
    elif change in ("probability", "label"):
        path = final / "test/predictions.csv"
        rows = read_csv(path)
        rows[0]["p_apple" if change == "probability" else "true_class"] = "nan" if change == "probability" else "banana"
        write_csv(path, rows)
    elif change == "manifest":
        with (completed_run / "dataset_manifest.csv").open("a") as f:
            f.write("\n")
    else:
        path = final / "test/metrics.json" if change == "metric" else completed_run / "summary.json"
        record = json.loads(path.read_text())
        if change == "metric":
            record["metrics"]["correct"] = 4
        else:
            record["results"]["32"]["test_accuracy"] = 0.8
        write_json(path, record)
    with pytest.raises(ValueError):
        evaluation.read_results(completed_run, (32,))


def test_prediction_loads_completed_l2_weights_and_accepts_rgb_conversion(completed_run, tmp_path):
    image = tmp_path / "photo.png"
    Image.new("L", (64, 48), 120).save(image)
    result = evaluation.predict_image(completed_run, image, 32, "cpu")
    assert set(result) == set(CLASSES)
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)


def test_completed_train_entrypoint_never_restarts_training(completed_run, monkeypatch):
    import train

    monkeypatch.setattr(train, "verify_python", lambda: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    monkeypatch.setattr(evaluation, "run_folder", lambda name: completed_run)
    monkeypatch.setattr(evaluation, "print_results", lambda results: None)
    monkeypatch.setattr(evaluation, "read_results", lambda folder, sizes: {})
    from src import training_pipeline

    monkeypatch.setattr(training_pipeline, "run_pipeline", lambda **kw: pytest.fail("Must not retrain"))
    assert train.main(["--sizes", "32"]) == 0


def test_run_name_cannot_escape_outputs(tmp_path):
    with pytest.raises(ValueError):
        evaluation.run_folder("../outside", tmp_path)
