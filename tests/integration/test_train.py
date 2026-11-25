"""End to end: synthetic dataset -> training -> checkpoints, logs and exact resume."""

import csv
import json
import os
import signal
import threading
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from chessnet.cli import app
from chessnet.config import MODEL_PRESETS, TrainConfig
from chessnet.model.checkpoint import load_checkpoint, load_network
from chessnet.model.resnet import ChessNet
from chessnet.train.loop import train

from helpers.datasets import make_dataset

SMOKE = TrainConfig(
    model="tiny",
    batch_size=64,
    max_steps=50,
    warmup_steps=5,
    log_every=10,
    val_every=25,
    val_batches=4,
    num_workers=0,
)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_dataset(tmp_path_factory.mktemp("data") / "shards", positions=2400, seed=0)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_smoke_training_run(dataset: Path, tmp_path: Path) -> None:
    result = train(SMOKE, dataset, tmp_path / "run", device="cpu")
    assert result.stopped_because == "completed"
    assert result.state.step == result.state.total_steps == 50
    assert result.state.epoch >= 1  # ~33 batches per epoch: the run crossed an epoch boundary

    run = tmp_path / "run"
    assert result.last_checkpoint == run / "checkpoints" / "step-000000050.pt"
    assert result.best_checkpoint == run / "best.pt"
    assert len(list((run / "checkpoints").glob("step-*.pt"))) <= SMOKE.keep_checkpoints
    checkpoint = load_checkpoint(result.last_checkpoint)
    assert checkpoint.step == 50
    assert checkpoint.extra["train_state"]["step"] == 50
    assert checkpoint.config["train"]["model"] == "tiny"
    assert checkpoint.dataset_manifest_hash is not None

    train_rows = read_rows(run / "metrics.csv")
    assert [row["step"] for row in train_rows] == ["10", "20", "30", "40", "50"]
    assert {"loss", "policy_loss", "value_loss", "top1", "lr", "grad_norm", "loss_scale"} <= set(
        train_rows[0]
    )
    val_rows = read_rows(run / "val_metrics.csv")
    assert [row["step"] for row in val_rows] == ["25", "50"]
    assert {"top1_white", "top1_black", "top1_colour_gap", "top1_opening", "value_mse"} <= set(
        val_rows[0]
    )
    assert list((run / "tb").glob("events.out.tfevents.*"))
    sessions = [json.loads(line) for line in (run / "runs.jsonl").read_text().splitlines()]
    assert len(sessions) == 1
    assert sessions[0]["device"] == "cpu"
    assert not sessions[0]["amp"]  # mixed precision is CUDA-only
    assert set(result.val_metrics) >= {"loss", "top1", "top3", "top5", "wdl_accuracy"}


def final_weights(run: Path) -> dict[str, torch.Tensor]:
    return load_checkpoint(run / "checkpoints" / "step-000000060.pt").model_state


@pytest.mark.parametrize(
    ("num_workers", "channels_last"),
    [
        (0, False),
        (0, True),
        pytest.param(2, False, marks=pytest.mark.slow),  # worker order is also unit-tested
    ],
)
def test_kill_and_resume_matches_an_uninterrupted_run(
    dataset: Path, tmp_path: Path, num_workers: int, channels_last: bool
) -> None:
    config = SMOKE.model_copy(
        update={
            "max_steps": 60,
            "log_every": 1,
            "deterministic": True,
            "num_workers": num_workers,
            "channels_last": channels_last,
        }
    )
    straight = train(config, dataset, tmp_path / "straight", device="cpu")
    first = train(config, dataset, tmp_path / "resumed", device="cpu", stop_at_step=30)
    assert first.stopped_because == "stop_at_step"
    assert first.state.step == 30
    assert first.state.epoch_batches > 0  # stopped in the middle of an epoch
    second = train(config, dataset, tmp_path / "resumed", device="cpu")
    assert second.state.step == straight.state.step == 60
    assert second.state.sessions == 2

    uninterrupted = read_rows(tmp_path / "straight" / "metrics.csv")
    interrupted = read_rows(tmp_path / "resumed" / "metrics.csv")
    assert [row["step"] for row in interrupted] == [str(step) for step in range(1, 61)]
    for a, b in zip(uninterrupted, interrupted, strict=True):
        for column in ("loss", "policy_loss", "value_loss", "lr"):
            assert float(a[column]) == pytest.approx(float(b[column]), rel=1e-6, abs=1e-9), (
                a["step"],
                column,
            )
    expected = final_weights(tmp_path / "straight")
    for name, tensor in final_weights(tmp_path / "resumed").items():
        torch.testing.assert_close(tensor, expected[name], rtol=1e-6, atol=1e-7)


@pytest.mark.slow
def test_speed_options_do_not_reach_the_checkpoint(dataset: Path, tmp_path: Path) -> None:
    """A compiled channels_last run resumes without either option, and the other way round."""
    fast = SMOKE.model_copy(update={"max_steps": 30, "compile": True, "channels_last": True})
    plain = fast.model_copy(update={"compile": False, "channels_last": False})
    keys = set(ChessNet(MODEL_PRESETS["tiny"]).state_dict())
    steps = []
    for config, stop_at_step in ((fast, 10), (plain, 20), (fast, None)):
        result = train(config, dataset, tmp_path / "run", device="cpu", stop_at_step=stop_at_step)
        assert result.last_checkpoint is not None
        assert set(load_checkpoint(result.last_checkpoint).model_state) == keys
        steps.append(result.state.step)
    assert steps == [10, 20, 30]
    assert result.state.sessions == 3
    assert load_network(result.last_checkpoint)(torch.zeros(1, 19, 8, 8)).policy.isfinite().all()


def test_sigterm_stops_cleanly_with_a_checkpoint(dataset: Path, tmp_path: Path) -> None:
    config = SMOKE.model_copy(update={"max_steps": 100_000, "val_every": 1_000_000})
    timer = threading.Timer(2.0, os.kill, args=(os.getpid(), signal.SIGTERM))
    handler_before = signal.getsignal(signal.SIGTERM)
    timer.start()
    try:
        result = train(config, dataset, tmp_path / "run", device="cpu")
    finally:
        timer.cancel()
    assert result.stopped_because == "signal"
    assert 0 < result.state.step < 100_000
    assert (
        result.last_checkpoint
        == tmp_path / "run" / "checkpoints" / f"step-{result.state.step:09d}.pt"
    )
    assert load_checkpoint(result.last_checkpoint).step == result.state.step
    assert signal.getsignal(signal.SIGTERM) == handler_before


def test_wall_time_budget_stops_cleanly(dataset: Path, tmp_path: Path) -> None:
    config = SMOKE.model_copy(update={"max_steps": 100_000, "val_every": 1_000_000})
    result = train(config, dataset, tmp_path / "run", device="cpu", max_wall_time=0.5)
    assert result.stopped_because == "max_wall_time"
    assert result.last_checkpoint is not None
    resumed = train(config, dataset, tmp_path / "run", device="cpu", max_wall_time=0.5)
    assert resumed.state.step > result.state.step  # picked up where it stopped


def test_resume_refuses_another_dataset(dataset: Path, tmp_path: Path) -> None:
    config = SMOKE.model_copy(update={"max_steps": 3})
    train(config, dataset, tmp_path / "run", device="cpu")
    other = make_dataset(tmp_path / "other", positions=800, seed=1)
    with pytest.raises(ValueError, match="different dataset"):
        train(config.model_copy(update={"max_steps": 6}), other, tmp_path / "run", device="cpu")


def test_resume_refuses_another_model(dataset: Path, tmp_path: Path) -> None:
    config = SMOKE.model_copy(update={"max_steps": 3})
    train(config, dataset, tmp_path / "run", device="cpu")
    with pytest.raises(ValueError, match="differs from the configured model"):
        train(config.model_copy(update={"model": "small"}), dataset, tmp_path / "run", device="cpu")


def test_explicit_fresh_start_ignores_checkpoints(dataset: Path, tmp_path: Path) -> None:
    config = SMOKE.model_copy(update={"max_steps": 3})
    train(config, dataset, tmp_path / "run", device="cpu")
    fresh = train(config, dataset, tmp_path / "run", device="cpu", resume="none")
    assert fresh.state.step == 3
    assert fresh.state.sessions == 1


def test_a_batch_larger_than_the_data_is_refused(dataset: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fewer positions than one batch"):
        train(SMOKE.model_copy(update={"batch_size": 10**6}), dataset, tmp_path / "run")


def test_cli_train(dataset: Path, tmp_path: Path) -> None:
    config = tmp_path / "train.yaml"
    config.write_text(
        "model: tiny\nbatch_size: 64\nmax_steps: 1000\nwarmup_steps: 1\nlog_every: 2\n"
        "val_every: 4\nval_batches: 2\nnum_workers: 0\n",
        encoding="utf-8",
    )
    common = ["train", "--config", str(config), "--data", str(dataset), "--device", "cpu"]
    result = CliRunner().invoke(app, [*common, "--out", str(tmp_path / "run"), "--max-steps", "4"])
    assert result.exit_code == 0, result.output
    assert "stopped at step 4 of 4 (completed)" in result.output
    bad = CliRunner().invoke(app, [*common, "--out", str(tmp_path / "x"), "--logger", "mlflow"])
    assert bad.exit_code != 0
    assert "unknown logger" in bad.output


def test_wall_clock_checkpoints_are_rotated(dataset: Path, tmp_path: Path) -> None:
    config = SMOKE.model_copy(
        update={
            "max_steps": 12,
            "checkpoint_minutes": 1e-9,  # due after every step
            "keep_checkpoints": 2,
            "val_every": 1_000_000,
        }
    )
    result = train(config, dataset, tmp_path / "run", device="cpu")
    names = sorted(p.name for p in (tmp_path / "run" / "checkpoints").glob("step-*.pt"))
    assert names == ["step-000000011.pt", "step-000000012.pt"]
    assert result.best_checkpoint == tmp_path / "run" / "best.pt"  # the final step validates
    assert result.last_checkpoint == tmp_path / "run" / "checkpoints" / "step-000000012.pt"
