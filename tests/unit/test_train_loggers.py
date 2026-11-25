import csv
import logging
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from chessnet.train.loggers import (
    TRAIN,
    VAL,
    ConsoleLogger,
    CSVLogger,
    TensorBoardLogger,
    WandbLogger,
    build_loggers,
)

TRAIN_ROW = {
    "loss": 3.5,
    "policy_loss": 3.0,
    "value_loss": 1.0,
    "top1": 0.125,
    "positions_per_second": 9000.0,
    "lr": 0.001,
    "epoch": 0.5,
    "eta_minutes": 42.0,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_csv_files_append_across_sessions(tmp_path: Path) -> None:
    CSVLogger(tmp_path).log(TRAIN, 10, TRAIN_ROW)
    CSVLogger(tmp_path).log(TRAIN, 20, {**TRAIN_ROW, "loss": 2.5})  # a resumed session
    CSVLogger(tmp_path).log(VAL, 20, {"loss": 2.9, "top1": 0.2})
    rows = read_csv(tmp_path / "metrics.csv")
    assert [(row["step"], row["loss"]) for row in rows] == [("10", "3.5"), ("20", "2.5")]
    assert read_csv(tmp_path / "val_metrics.csv") == [{"step": "20", "loss": "2.9", "top1": "0.2"}]


def test_console_lines(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="chessnet.train"):
        console = ConsoleLogger()
        console.log(TRAIN, 1200, TRAIN_ROW)
        console.log(VAL, 1200, {"top1": 0.3125, "loss": 2.0})
        console.close()
    train_line, val_block = caplog.messages
    assert "step 1,200" in train_line
    assert "top1 12.5%" in train_line
    assert "9,000 pos/s" in train_line
    assert "ETA 42 min" in train_line
    assert val_block.splitlines()[0] == "validation at step 1,200:"
    assert "0.3125" in val_block


def test_tensorboard_writes_event_files(tmp_path: Path) -> None:
    logger = TensorBoardLogger(tmp_path / "tb")
    logger.log(TRAIN, 1, TRAIN_ROW)
    logger.close()
    assert list((tmp_path / "tb").glob("events.out.tfevents.*"))


def test_build_loggers_without_wandb(tmp_path: Path) -> None:
    loggers = build_loggers(tmp_path)
    loggers.log(TRAIN, 5, TRAIN_ROW)
    loggers.close()
    assert [type(logger).__name__ for logger in loggers.loggers] == [
        "ConsoleLogger",
        "CSVLogger",
        "TensorBoardLogger",
    ]
    assert (tmp_path / "metrics.csv").exists()


def test_wandb_without_api_key_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with caplog.at_level(logging.WARNING, logger="chessnet.train"):
        logger = WandbLogger(tmp_path, {}, run_id=None)
    logger.log(TRAIN, 1, TRAIN_ROW)
    logger.close()
    assert "WANDB_API_KEY is not set" in caplog.text


class _FakeRun:
    def __init__(self, fail_on_log: bool) -> None:
        self.fail_on_log = fail_on_log
        self.logged: list[tuple[dict[str, float], int]] = []
        self.finished = False

    def log(self, data: dict[str, float], step: int) -> None:
        if self.fail_on_log:
            raise ConnectionError("server unreachable")
        self.logged.append((data, step))

    def finish(self) -> None:
        self.finished = True


def fake_wandb(monkeypatch: pytest.MonkeyPatch, run: _FakeRun | Exception) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def init(**kwargs: Any) -> _FakeRun:
        calls.append(kwargs)
        if isinstance(run, Exception):
            raise run
        return run

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=init))
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    return calls


def test_wandb_logs_through_the_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _FakeRun(fail_on_log=False)
    calls = fake_wandb(monkeypatch, run)
    logger = WandbLogger(tmp_path, {"model": "tiny"}, run_id="chessnet-abc")
    logger.log(VAL, 7, {"top1": 0.5})
    logger.close()
    assert calls[0]["project"] == "chessnet"
    assert calls[0]["id"] == "chessnet-abc"
    assert run.logged == [({"val/top1": 0.5}, 7)]
    assert run.finished


@pytest.mark.parametrize("failure", ["init", "log"])
def test_wandb_failures_never_stop_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: str,
) -> None:
    run = _FakeRun(fail_on_log=failure == "log")
    fake_wandb(monkeypatch, RuntimeError("no network") if failure == "init" else run)
    with caplog.at_level(logging.WARNING, logger="chessnet.train"):
        logger = WandbLogger(tmp_path, {}, run_id=None)
        logger.log(TRAIN, 1, TRAIN_ROW)
        logger.log(TRAIN, 2, TRAIN_ROW)  # disabled after the first failure
        logger.close()
    assert "W&B disabled" in caplog.text
    assert not run.finished


def test_wandb_is_added_on_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_wandb(monkeypatch, _FakeRun(fail_on_log=False))
    loggers = build_loggers(tmp_path, wandb=True, config={"a": 1}, run_id="x")
    assert type(loggers.loggers[-1]).__name__ == "WandbLogger"
    loggers.close()
