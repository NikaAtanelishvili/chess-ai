"""The training loop: mixed precision, wall-clock checkpoints, exact resume, graceful stop.

* fp16 autocast with a GradScaler on CUDA (T4s handle bf16 poorly); plain fp32 on the CPU.
* AdamW (default) or Nesterov SGD, linear warmup then cosine decay, gradient clipping.
* Validation on a fixed subset of the val split every ``val_every`` steps.
* Checkpoints every ``checkpoint_minutes`` of wall time and at the end of each epoch (the
  last ``keep_checkpoints`` are kept), plus ``best.pt`` by validation loss.
* Resume restores weights, optimizer, scheduler, scaler, RNG and the position in the data,
  so an interrupted run continues exactly where it stopped.
* SIGTERM/SIGINT or ``max_wall_time`` stop the run cleanly after the current step, with a
  final checkpoint.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import signal
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from types import FrameType
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from chessnet import __version__
from chessnet.config import MODEL_PRESETS, TrainConfig
from chessnet.data.dataset import ShardDataset, make_loader
from chessnet.data.shards import Split, manifest_hash, read_manifest
from chessnet.encoding.packed import PackedBatch, unpack_batch_torch
from chessnet.model.checkpoint import (
    Checkpoint,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from chessnet.model.resnet import ChessNet
from chessnet.provenance import git_commit
from chessnet.train.loggers import TRAIN, VAL, MetricLogger, build_loggers
from chessnet.train.losses import compute_losses
from chessnet.train.metrics import MetricAccumulator

log = logging.getLogger("chessnet.train")

CHECKPOINT_DIR = "checkpoints"
BEST_CHECKPOINT = "best.pt"
_DURATION = re.compile(
    r"^(?:(?P<h>\d+(?:\.\d+)?)h)?(?:(?P<m>\d+(?:\.\d+)?)m)?(?:(?P<s>\d+(?:\.\d+)?)s)?$"
)


def parse_duration(text: str) -> float:
    """Seconds in a duration like ``11h15m``, ``90m``, ``45s`` or ``3600``."""
    text = text.strip().lower()
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    match = _DURATION.fullmatch(text)
    if not text or match is None or not any(match.groupdict().values()):
        raise ValueError(f"not a duration: {text!r} (examples: 11h15m, 90m, 45s)")
    hours, minutes, seconds = (float(match[group] or 0) for group in ("h", "m", "s"))
    return hours * 3600 + minutes * 60 + seconds


def warmup_cosine(step: int, *, warmup: int, total: int, min_ratio: float) -> float:
    """LR multiplier: linear warmup over ``warmup`` steps, then cosine decay to ``min_ratio``."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min((step - warmup) / max(total - warmup, 1), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass
class TrainState:
    """Progress that a checkpoint must restore besides the model and optimizer states."""

    step: int = 0
    total_steps: int = 0
    epoch: int = 0
    epoch_batches: int = 0  # batches of the current epoch already trained on
    best_val_loss: float = math.inf
    sessions: int = 0


@dataclass(frozen=True)
class TrainResult:
    state: TrainState
    stopped_because: str  # "completed", "max_wall_time", "signal" or "stop_at_step"
    last_checkpoint: Path | None
    best_checkpoint: Path | None
    val_metrics: dict[str, float] = field(default_factory=dict)


class StopRequest:
    """Collects reasons to stop early: SIGTERM/SIGINT, a wall-clock budget, a step limit."""

    def __init__(self, *, max_wall_time: float | None, started: float) -> None:
        self.max_wall_time = max_wall_time
        self.started = started
        self.signal_name: str | None = None

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        self.signal_name = signal.Signals(signum).name
        log.warning("received %s: stopping after the current step", self.signal_name)

    @contextmanager
    def handling_signals(self) -> Iterator[StopRequest]:
        """Install handlers for SIGTERM and SIGINT (main thread only); restore them after."""
        previous = {
            sig: signal.signal(sig, self._handle) for sig in (signal.SIGTERM, signal.SIGINT)
        }
        try:
            yield self
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def reason(self, now: float, step: int, stop_at_step: int | None) -> str | None:
        if self.signal_name is not None:
            return "signal"
        # A deadline by addition: ``now - started`` can round to just below the budget.
        if self.max_wall_time is not None and now >= self.started + self.max_wall_time:
            return "max_wall_time"
        if stop_at_step is not None and step >= stop_at_step:
            return "stop_at_step"
        return None


def prepare_network(network: ChessNet, *, channels_last: bool, compile: bool) -> nn.Module:
    """Apply the speed options; returns the module to call (compiled or the network itself).

    Weights stay in ``network``, so checkpoints never depend on these options.
    """
    if channels_last:
        network.to(memory_format=torch.channels_last)
    return cast(nn.Module, torch.compile(network)) if compile else network


def model_input(planes: torch.Tensor, *, channels_last: bool) -> torch.Tensor:
    """Input planes in the memory format the network expects."""
    return planes.contiguous(memory_format=torch.channels_last) if channels_last else planes


def make_optimizer(config: TrainConfig, network: nn.Module) -> torch.optim.Optimizer:
    """AdamW or Nesterov SGD; biases and normalisation weights get no weight decay."""
    decay = [p for p in network.parameters() if p.requires_grad and p.ndim > 1]
    no_decay = [p for p in network.parameters() if p.requires_grad and p.ndim <= 1]
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if config.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=config.lr, momentum=config.momentum, nesterov=True)
    return torch.optim.AdamW(groups, lr=config.lr)


def find_checkpoint(resume: str, out_dir: Path, resume_dirs: Sequence[Path]) -> Path | None:
    """The checkpoint to resume from: ``none``, an explicit path, or ``auto`` (the newest
    rolling checkpoint in ``out_dir`` or any of ``resume_dirs``)."""
    if resume == "none":
        return None
    if resume != "auto":
        path = Path(resume)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path
    candidates = list((out_dir / CHECKPOINT_DIR).glob("step-*.pt"))
    for directory in resume_dirs:
        if directory.is_dir():
            candidates += directory.rglob("step-*.pt")
    return max(candidates, key=lambda p: (int(p.stem.removeprefix("step-")), str(p)), default=None)


#: torch functions that run on MKL's vector math on the CPU, and that training may use.
_VECTOR_MATH = (
    torch.sqrt, torch.rsqrt, torch.exp, torch.expm1, torch.log, torch.log1p, torch.tanh,
    torch.sigmoid, torch.erf, torch.sin, torch.cos,
)  # fmt: skip


def warm_up_vector_math() -> None:
    """Run torch's MKL vector-math kernels once, on this thread alone.

    MKL sets them up on first use. When that first use is split across threads (torch hands
    2,048-element chunks to its threads), one thread can compute at reduced accuracy: the
    first square root Adam took was off by up to 3e-4 relative in 1 to 7 of every 8 fresh
    processes on a 2-core laptop, so a run could differ from its resumed copy. 1,024 values
    stay below the chunk size, so these calls never leave the calling thread.
    """
    values = torch.full((1024,), 0.5)
    for function in _VECTOR_MATH:
        function(values)


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


class _Checkpointer:
    """Writes rolling, end-of-epoch, final and best checkpoints for one run."""

    def __init__(
        self,
        out_dir: Path,
        keep: int,
        *,
        network: ChessNet,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: torch.amp.GradScaler,
        config: dict[str, Any],
        manifest_hash: str,
    ) -> None:
        self.directory = out_dir / CHECKPOINT_DIR
        self.directory.mkdir(parents=True, exist_ok=True)
        self.best_path = out_dir / BEST_CHECKPOINT
        self.keep = keep
        self.network = network
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.config = config
        self.manifest_hash = manifest_hash
        self.last: Path | None = None

    def save(self, state: TrainState, *, metrics: dict[str, float], best: bool = False) -> Path:
        path = self.best_path if best else self.directory / f"step-{state.step:09d}.pt"
        save_checkpoint(
            path,
            self.network,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            step=state.step,
            epoch=state.epoch,
            config=self.config,
            metrics=metrics,
            dataset_manifest_hash=self.manifest_hash,
            extra={"train_state": asdict(state)},
        )
        if not best:
            self.last = path
            rolling = sorted(self.directory.glob("step-*.pt"))
            for old in rolling[: max(len(rolling) - self.keep, 0)]:
                old.unlink()
        return path


def _validate(
    network: nn.Module,
    batches: Sequence[PackedBatch],
    device: torch.device,
    use_amp: bool,
    config: TrainConfig,
) -> dict[str, float]:
    accumulator = MetricAccumulator()
    network.eval()
    with torch.inference_mode():
        for cpu_batch in batches:
            batch = cpu_batch.to(device, non_blocking=True)
            planes = model_input(unpack_batch_torch(batch), channels_last=config.channels_last)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                output = network(planes)
            accumulator.update(
                output,
                batch.move,
                batch.wdl,
                batch.ply,
                label_smoothing=config.label_smoothing,
                value_weight=config.value_loss_weight,
            )
    network.train()
    return accumulator.compute()


def _restore(
    checkpoint: Checkpoint,
    *,
    network: ChessNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    expected_manifest_hash: str,
) -> TrainState:
    if checkpoint.model_config != network.config:
        raise ValueError(
            f"checkpoint model {checkpoint.model_config} differs from the configured model "
            f"{network.config}"
        )
    if checkpoint.dataset_manifest_hash != expected_manifest_hash:
        raise ValueError(
            "the checkpoint was trained on a different dataset (manifest hash mismatch); "
            "resume with the same data or start a new run"
        )
    restore_training_state(
        checkpoint,
        network,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        restore_rng=True,
    )
    return TrainState(**checkpoint.extra["train_state"])


def train(
    config: TrainConfig,
    data_dir: Path,
    out_dir: Path,
    *,
    resume: str = "auto",
    resume_dirs: Sequence[Path] = (),
    max_wall_time: float | None = None,
    wandb: bool = False,
    device: str | None = None,
    stop_at_step: int | None = None,
) -> TrainResult:
    """Train (or continue training) a network on a dataset directory.

    ``resume``: ``auto`` (newest checkpoint in ``out_dir/checkpoints`` or ``resume_dirs``),
    ``none``, or a checkpoint path. ``max_wall_time`` (seconds) and ``stop_at_step`` stop
    the run early with a final checkpoint, as SIGTERM and SIGINT do.
    """
    started = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = config.amp and torch_device.type == "cuda"
    warm_up_vector_math()  # before any parallel CPU work, or a resumed run can differ
    _seed_everything(config.seed, config.deterministic)

    read_manifest(data_dir)  # refuses data from another encoding version
    dataset_hash = manifest_hash(data_dir)
    train_set = ShardDataset(
        data_dir, Split.TRAIN, batch_size=config.batch_size, shuffle=True, seed=config.seed
    )
    steps_per_epoch = train_set.num_batches()
    if steps_per_epoch == 0:
        raise ValueError("the training split holds fewer positions than one batch")
    val_set = ShardDataset(
        data_dir, Split.VAL, batch_size=config.batch_size, shuffle=False, drop_last=False
    )
    val_batches = list(islice(iter(val_set), config.val_batches))

    network = ChessNet(MODEL_PRESETS[config.model]).to(torch_device)
    # Before the optimizer exists, so its state takes the parameters' memory format.
    model = prepare_network(network, channels_last=config.channels_last, compile=config.compile)
    optimizer = make_optimizer(config, network)
    state = TrainState(
        total_steps=config.max_steps or max(1, round(config.epochs * steps_per_epoch))
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: warmup_cosine(
            step,
            warmup=config.warmup_steps,
            total=state.total_steps,
            min_ratio=config.min_lr_ratio,
        ),
    )
    scaler = torch.amp.GradScaler(torch_device.type, enabled=use_amp)

    checkpoint_path = find_checkpoint(resume, out_dir, resume_dirs)
    if checkpoint_path is not None:
        state = _restore(
            load_checkpoint(checkpoint_path, map_location=torch_device),
            network=network,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_manifest_hash=dataset_hash,
        )
        log.info("resumed from %s at step %s", checkpoint_path, f"{state.step:,}")
    state.sessions += 1

    run_config = {"train": config.model_dump(), "data": str(data_dir)}
    session = {
        "session": state.sessions,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "resumed_from": str(checkpoint_path) if checkpoint_path else None,
        "start_step": state.step,
        "total_steps": state.total_steps,
        "device": str(torch_device),
        "amp": use_amp,
        "chessnet_version": __version__,
        "git_commit": git_commit(),
        "dataset_manifest_hash": dataset_hash,
        "config": run_config,
    }
    with (out_dir / "runs.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(session) + "\n")

    loggers: MetricLogger = build_loggers(
        out_dir, wandb=wandb, config=session, run_id=f"chessnet-{dataset_hash[:8]}"
    )
    checkpointer = _Checkpointer(
        out_dir,
        config.keep_checkpoints,
        network=network,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config=run_config,
        manifest_hash=dataset_hash,
    )
    stop = StopRequest(max_wall_time=max_wall_time, started=started)
    stopped_because = "completed"
    val_metrics: dict[str, float] = {}
    last_checkpoint_time = time.monotonic()

    window_start, window_steps = time.monotonic(), 0
    sums = torch.zeros(4, device=torch_device)  # loss, policy, value, top1 over the window
    network.train()
    with stop.handling_signals():
        while state.step < state.total_steps and stopped_because == "completed":
            train_set.set_epoch(state.epoch)
            train_set.start_after(state.epoch_batches, num_workers=config.num_workers)
            loader = make_loader(train_set, num_workers=config.num_workers)
            for cpu_batch in loader:
                batch = cpu_batch.to(torch_device, non_blocking=True)
                planes = model_input(unpack_batch_torch(batch), channels_last=config.channels_last)
                with torch.autocast(torch_device.type, dtype=torch.float16, enabled=use_amp):
                    output = model(planes)
                    losses = compute_losses(
                        output,
                        batch.move,
                        batch.wdl,
                        value_weight=config.value_loss_weight,
                        label_smoothing=config.label_smoothing,
                    )
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(losses.total).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                state.step += 1
                state.epoch_batches += 1
                window_steps += 1
                with torch.no_grad():  # kept on the device: no GPU sync until a log step
                    top1 = (output.policy.argmax(dim=1) == batch.move).float().mean()
                    sums += torch.stack(
                        [losses.total.detach(), losses.policy.detach(), losses.value.detach(), top1]
                    )

                now = time.monotonic()
                if state.step % config.log_every == 0 or state.step == state.total_steps:
                    elapsed = max(now - window_start, 1e-9)
                    loss, policy, value, top1 = (sums / window_steps).tolist()
                    rate = window_steps / elapsed
                    metrics = {
                        "loss": loss,
                        "policy_loss": policy,
                        "value_loss": value,
                        "top1": top1,
                        "lr": optimizer.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm),
                        "loss_scale": scaler.get_scale() if use_amp else 1.0,
                        "positions_per_second": rate * config.batch_size,
                        "epoch": state.epoch + state.epoch_batches / steps_per_epoch,
                        "eta_minutes": (state.total_steps - state.step) / rate / 60,
                    }
                    if torch_device.type == "cuda":
                        metrics["gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
                    loggers.log(TRAIN, state.step, metrics)
                    window_start, window_steps = now, 0
                    sums.zero_()

                if val_batches and (
                    state.step % config.val_every == 0 or state.step == state.total_steps
                ):
                    val_metrics = _validate(network, val_batches, torch_device, use_amp, config)
                    loggers.log(VAL, state.step, val_metrics)
                    if val_metrics["loss"] < state.best_val_loss:
                        state.best_val_loss = val_metrics["loss"]
                        checkpointer.save(state, best=True, metrics=val_metrics)

                if now - last_checkpoint_time >= config.checkpoint_minutes * 60:
                    checkpointer.save(state, metrics=val_metrics)
                    last_checkpoint_time = now

                reason = stop.reason(now, state.step, stop_at_step)
                if reason is not None:
                    stopped_because = reason
                    break
                if state.step >= state.total_steps:
                    break
            else:  # the loader ran dry: the epoch is complete
                if state.epoch_batches == 0:
                    raise ValueError("an epoch yielded no batches")
                state.epoch += 1
                state.epoch_batches = 0
                checkpointer.save(state, metrics=val_metrics)
                last_checkpoint_time = time.monotonic()

    if checkpointer.last is None or checkpointer.last.stem != f"step-{state.step:09d}":
        checkpointer.save(state, metrics=val_metrics)
    loggers.close()
    log.info(
        "stopped at step %s of %s (%s); last checkpoint %s",
        f"{state.step:,}",
        f"{state.total_steps:,}",
        stopped_because,
        checkpointer.last,
    )
    best = checkpointer.best_path if checkpointer.best_path.exists() else None
    return TrainResult(state, stopped_because, checkpointer.last, best, val_metrics)
