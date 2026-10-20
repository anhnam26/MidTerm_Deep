import hashlib
import json
import os

import pytest
import torch
from torch.utils.data import TensorDataset

from src import training_runtime as runtime
from src.common import read_csv


def tiny_data():
    generator = torch.Generator().manual_seed(14)
    return TensorDataset(torch.rand(10, 3, 32, 32, generator=generator), torch.arange(10) % 5)


def config(**overrides):
    return runtime.TrainConfig(
        size=32, epochs=2, batch_size=5, workers=0, device="cpu", mixup_probability=0, **overrides
    )


def test_completed_epochs_overwrite_two_checkpoints_and_export_metrics(tmp_path, capsys):
    data = tiny_data()
    summary = runtime.train_job(config(), tmp_path, data, data, {"fixture": True})
    assert summary["completed_epochs"] == 2
    assert {p.name for p in tmp_path.glob("*.pt")} == {"best.pt", "last.pt"}
    last = torch.load(tmp_path / "last.pt", weights_only=True)
    best = torch.load(tmp_path / "best.pt", weights_only=True)
    assert last["epoch"] == 2 and len(last["history"]) == 2
    assert last["optimizer"]["state"] and "rng" in last and "scaler" in last
    assert best["epoch"] == last["best_epoch"]
    assert all(torch.equal(v, last["best_weights"][k]) for k, v in best["model"].items())
    history = read_csv(tmp_path / "epochs.csv")
    assert len(history) == 2 and "val_macro_f1" in history[0]
    metrics = json.loads((tmp_path / "epochs.json").read_text())
    assert len(metrics[0]["val"]["confusion_matrix"]) == 5
    assert len(metrics[0]["val"]["per_class"]) == 5
    assert len((tmp_path / "batches.jsonl").read_text().splitlines()) >= 4
    screen = capsys.readouterr().out
    lines = screen.splitlines()
    assert len([line for line in lines if "EPOCH " in line]) == 2
    assert "EPOCH 2/2" in screen
    assert "batch=" not in screen and " completed loss=" not in screen
    assert "samples=" not in screen and "CONFIG " not in screen
    details = (tmp_path / "console.log").read_text()
    assert "batch=" in details and "samples=" in details
    before = (tmp_path / "last.pt").read_bytes()
    runtime.train_job(config(), tmp_path, data, data, {"fixture": True})
    assert (tmp_path / "last.pt").read_bytes() == before


def test_partial_epoch_is_not_checkpointed_and_resume_matches_uninterrupted(tmp_path, monkeypatch, capsys):
    data = tiny_data()
    interrupted = tmp_path / "resumed"
    original = runtime.run_epoch

    def fail_in_second_validation(*args, **kwargs):
        if kwargs["epoch"] == 2 and kwargs["phase"] == "val":
            raise KeyboardInterrupt("simulated interruption during validation")
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "run_epoch", fail_in_second_validation)
    with pytest.raises(KeyboardInterrupt):
        runtime.train_job(config(), interrupted, data, data, {"fixture": True})
    interrupted_output = capsys.readouterr().out
    assert interrupted_output.count("EPOCH ") == 1
    assert "EPOCH 2/2" not in interrupted_output
    assert "batch=" not in interrupted_output
    first = torch.load(interrupted / "last.pt", weights_only=True)
    assert first["epoch"] == 1 and len(first["history"]) == 1
    monkeypatch.setattr(runtime, "run_epoch", original)
    runtime.train_job(config(), interrupted, data, data, {"fixture": True})
    resumed_output = capsys.readouterr().out
    assert "RESUME" in resumed_output and "epoch=1 -> next_epoch=2" in resumed_output
    assert "EPOCH 1/2" not in resumed_output and "EPOCH 2/2" in resumed_output
    runtime.train_job(config(), tmp_path / "reference", data, data, {"fixture": True})
    resumed = torch.load(interrupted / "last.pt", weights_only=True)
    reference = torch.load(tmp_path / "reference" / "last.pt", weights_only=True)
    assert all(torch.equal(v, reference["model"][k]) for k, v in resumed["model"].items())
    assert len(read_csv(interrupted / "epochs.csv")) == 2


def test_missing_last_cannot_silently_restart_a_completed_run(tmp_path):
    data = tiny_data()
    runtime.train_job(config(), tmp_path, data, data, {"fixture": True})
    best = (tmp_path / "best.pt").read_bytes()
    (tmp_path / "last.pt").unlink()
    with pytest.raises(ValueError, match="Missing last.pt"):
        runtime.train_job(config(), tmp_path, data, data, {"fixture": True})
    assert not (tmp_path / "last.pt").exists()
    assert (tmp_path / "best.pt").read_bytes() == best


def test_resume_refuses_changed_settings_or_provenance(tmp_path):
    data = tiny_data()
    runtime.train_job(config(), tmp_path, data, data, {"fixture": True})
    digest = hashlib.sha256((tmp_path / "last.pt").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="changed"):
        runtime.train_job(config(lr=0.007), tmp_path, data, data, {"fixture": True})
    with pytest.raises(ValueError, match="changed"):
        runtime.train_job(config(), tmp_path, data, data, {"fixture": False})
    assert hashlib.sha256((tmp_path / "last.pt").read_bytes()).hexdigest() == digest


def test_final_refit_checkpoint_selection_never_uses_training_accuracy(tmp_path, capsys):
    data = tiny_data()
    runtime.train_job(config(), tmp_path, data, None, {"fixture": True}, selection_kind="ema")
    screen = capsys.readouterr().out
    assert screen.count("EPOCH ") == 2 and "batch=" not in screen
    last = torch.load(tmp_path / "last.pt", weights_only=True)
    assert last["best_epoch"] == 2
    assert last["selection_rule"] == "fixed_final_epoch"
    assert last["history"][-1]["val"] is None
    best = torch.load(tmp_path / "best.pt", weights_only=True)
    assert all(torch.equal(v, last["ema"][k]) for k, v in best["model"].items())


def test_nonfinite_loss_cannot_advance_last_checkpoint(tmp_path):
    bad = TensorDataset(torch.full((5, 3, 32, 32), float("nan")), torch.arange(5))
    with pytest.raises(FloatingPointError):
        runtime.train_job(config(), tmp_path, bad, tiny_data(), {"fixture": True})
    assert not (tmp_path / "last.pt").exists()


def test_atomic_write_failure_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    runtime.atomic_save({"epoch": 1}, path)
    original = path.read_bytes()

    def broken_save(value, handle):
        handle.write(b"partial write")
        raise OSError("disk full fixture")

    monkeypatch.setattr(runtime.torch, "save", broken_save)
    with pytest.raises(OSError, match="disk full"):
        runtime.atomic_save({"epoch": 2}, path)
    assert path.read_bytes() == original
    assert not path.with_suffix(".pt.tmp").exists()


def test_best_does_not_regress_when_validation_declines(tmp_path, monkeypatch):
    original = runtime.run_epoch

    def decreasing_validation(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs["phase"] != "train":
            result[0]["accuracy"] = 0.9 if kwargs["epoch"] == 1 else 0.1
        return result

    monkeypatch.setattr(runtime, "run_epoch", decreasing_validation)
    data = tiny_data()
    runtime.train_job(config(), tmp_path, data, data, {"fixture": True})
    assert torch.load(tmp_path / "last.pt", weights_only=True)["epoch"] == 2
    assert torch.load(tmp_path / "best.pt", weights_only=True)["epoch"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP overflow behavior")
def test_amp_overflow_skips_one_update_and_recovers(tmp_path):
    from copy import deepcopy
    from torch.utils.data import DataLoader

    model = torch.nn.Linear(3, 5).cuda()
    ema = deepcopy(model).requires_grad_(False)
    calls = []

    def overflow_once(gradient):
        calls.append(1)
        return gradient * float("inf") if len(calls) == 1 else gradient

    model.weight.register_hook(overflow_once)
    data = TensorDataset(torch.rand(10, 3), torch.arange(10) % 5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda")
    log = runtime.RunLog(tmp_path)
    try:
        metrics, _, updates = runtime.run_epoch(
            model,
            DataLoader(data, batch_size=5),
            torch.device("cuda"),
            config(),
            log,
            epoch=1,
            phase="train",
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
        )
    finally:
        log.close()
    assert metrics["amp_skipped_steps"] == 1 and updates == 1
    assert all(torch.isfinite(p).all() for p in model.parameters())


class WorkerIdentityDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return os.getpid(), index


@pytest.mark.parametrize("workers", [0, 1])
def test_validation_loader_reuses_workers_and_preserves_order(workers):
    data = runtime.loader(WorkerIdentityDataset(), 4, workers, False, 2026, False, persistent=True)
    first = [(pids.tolist(), indices.tolist()) for pids, indices in data]
    second = [(pids.tolist(), indices.tolist()) for pids, indices in data]
    assert first == second
    assert [index for _, indices in first for index in indices] == list(range(8))
