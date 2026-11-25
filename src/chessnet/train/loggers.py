"""Metric loggers behind one interface: console, CSV and TensorBoard always; W&B on request.

W&B is imported lazily and every W&B failure is caught: a missing package, a missing API
key or an unreachable server must never stop a training run.
"""

from __future__ import annotations

import csv
import logging
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Protocol

log = logging.getLogger("chessnet.train")

#: Kinds of metric rows: training-progress rows and validation rows.
TRAIN = "train"
VAL = "val"


class MetricLogger(Protocol):
    def log(self, kind: str, step: int, metrics: Mapping[str, float]) -> None: ...

    def close(self) -> None: ...


def _format(value: float) -> str:
    return f"{value:.4g}"


class ConsoleLogger:
    """One line per training log step and a block per validation (Kaggle shows stdout live)."""

    def log(self, kind: str, step: int, metrics: Mapping[str, float]) -> None:
        if kind == TRAIN:
            m = metrics
            parts = [
                f"step {step:,}",
                f"epoch {m.get('epoch', 0):.2f}",
                f"loss {m.get('loss', math.nan):.4f}",
                f"(policy {m.get('policy_loss', math.nan):.4f}, "
                f"value {m.get('value_loss', math.nan):.4f})",
                f"top1 {m.get('top1', math.nan):.1%}",
                f"{m.get('positions_per_second', 0):,.0f} pos/s",
                f"lr {m.get('lr', math.nan):.2e}",
            ]
            if "eta_minutes" in m:
                parts.append(f"ETA {m['eta_minutes']:,.0f} min")
            log.info(" | ".join(parts))
        else:
            lines = [f"validation at step {step:,}:"]
            lines += [f"  {name:<20} {_format(value)}" for name, value in sorted(metrics.items())]
            log.info("\n".join(lines))

    def close(self) -> None:
        pass


class CSVLogger:
    """Appends rows to ``metrics.csv`` (training) and ``val_metrics.csv`` (validation).

    Resuming appends to the existing files; the columns are fixed by each file's first row.
    """

    FILES: ClassVar[dict[str, str]] = {TRAIN: "metrics.csv", VAL: "val_metrics.csv"}

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def log(self, kind: str, step: int, metrics: Mapping[str, float]) -> None:
        path = self.directory / self.FILES[kind]
        row = {"step": step, **metrics}
        if path.exists() and path.stat().st_size > 0:
            with path.open(encoding="utf-8", newline="") as file:
                fields = next(csv.reader(file))
            write_header = False
        else:
            fields = list(row)
            write_header = True
        with path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore", restval="")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def close(self) -> None:
        pass


class TensorBoardLogger:
    """Scalars under ``<kind>/<name>`` in ``<out>/tb`` (needs the ``train`` extra)."""

    def __init__(self, directory: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter  # optional dependency, imported late

        self._writer = SummaryWriter(log_dir=str(directory))

    def log(self, kind: str, step: int, metrics: Mapping[str, float]) -> None:
        for name, value in metrics.items():
            self._writer.add_scalar(f"{kind}/{name}", value, step)
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


class WandbLogger:
    """Weights & Biases, best effort: any failure disables it with a warning."""

    def __init__(self, directory: Path, config: Mapping[str, Any], run_id: str | None) -> None:
        self._run: Any = None
        if not os.environ.get("WANDB_API_KEY"):
            log.warning("W&B requested but WANDB_API_KEY is not set; continuing without W&B")
            return
        try:
            import wandb  # optional, imported only when requested

            self._run = wandb.init(
                project="chessnet",
                dir=str(directory),
                config=dict(config),
                id=run_id,
                resume="allow",
            )
        except Exception as error:  # W&B must never stop a training run
            log.warning("W&B disabled: %s", error)
            self._run = None

    def log(self, kind: str, step: int, metrics: Mapping[str, float]) -> None:
        if self._run is None:
            return
        try:
            self._run.log({f"{kind}/{name}": value for name, value in metrics.items()}, step=step)
        except Exception as error:
            log.warning("W&B disabled: %s", error)
            self._run = None

    def close(self) -> None:
        if self._run is not None:
            try:
                self._run.finish()
            except Exception as error:
                log.warning("W&B could not finish cleanly: %s", error)


class MultiLogger:
    """Sends every row to all loggers."""

    def __init__(self, loggers: Sequence[MetricLogger]) -> None:
        self.loggers = list(loggers)

    def log(self, kind: str, step: int, metrics: Mapping[str, float]) -> None:
        for logger in self.loggers:
            logger.log(kind, step, metrics)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()


def build_loggers(
    out_dir: Path,
    *,
    wandb: bool = False,
    config: Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> MultiLogger:
    """Console, CSV and TensorBoard loggers, plus W&B if requested."""
    loggers: list[MetricLogger] = [
        ConsoleLogger(),
        CSVLogger(out_dir),
        TensorBoardLogger(out_dir / "tb"),
    ]
    if wandb:
        loggers.append(WandbLogger(out_dir, config or {}, run_id))
    return MultiLogger(loggers)
