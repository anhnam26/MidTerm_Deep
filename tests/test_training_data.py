import pytest
import torch

from src import training_data
from src.model import FruitLite
from src.profile import profile


@pytest.mark.parametrize("size", [32, 224])
def test_l2_budget_and_gradients(size):
    model = FruitLite(variant="L2")
    measured = profile(model, size)
    assert measured["eligible"]
    assert measured["output_shape"] == [1, 5]
    assert measured["learnable_parameters"] > 600_000
    logits = model(torch.randn(2, 3, size, size))
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 4])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def fixture_rows():
    rows = []
    for fruit in training_data.CLASSES:
        for i in range(20):
            rows.append({"id": f"{fruit}_{i}", "class": fruit, "split": "train",
                         "group_id": f"legacy_train_{fruit}", "source_identity": f"{i}.jpg"})
        rows.append({"id": f"{fruit}_test", "class": fruit, "split": "test",
                     "group_id": f"legacy_test_{fruit}"})
    return rows


def test_split_is_reproducible_test_untouched_and_unknown_groups_disclosed():
    rows = fixture_rows()
    train, val, report = training_data.development_split(rows, .2, 2026)
    a, b, _ = training_data.development_split(list(reversed(rows)), .2, 2026)
    assert {r["id"] for r in a} == {r["id"] for r in train}
    assert {r["id"] for r in b} == {r["id"] for r in val}
    assert len(train) == 80 and len(val) == 20
    assert all(r["split"] == "train" for r in train + val)
    assert {r["id"] for r in train}.isdisjoint(r["id"] for r in val)
    assert report["test_reserved"] == 5
    assert report["unverified_capture_images"] == 100
    assert not report["physical_independence_verified"]


def test_specimen_views_stay_together():
    rows = fixture_rows()
    for row in rows:
        if row["class"] == "pineapple" and row["split"] == "train":
            row["group_id"] = f"pqc500_specimen_{int(row['id'].split('_')[-1]) // 2}"
    train, val, report = training_data.development_split(rows, .2, 7)
    train_groups = {r["group_id"] for r in train if r["class"] == "pineapple"}
    val_groups = {r["group_id"] for r in val if r["class"] == "pineapple"}
    assert train_groups.isdisjoint(val_groups)
    assert report["unverified_capture_images"] == 80


def test_single_physical_group_cannot_be_split_to_make_validation():
    rows = fixture_rows()
    for row in rows:
        if row["class"] == "pineapple" and row["split"] == "train":
            row["group_id"] = "pqc500_specimen_only_one"
    with pytest.raises(ValueError, match="groups"):
        training_data.development_split(rows, .2, 7)
