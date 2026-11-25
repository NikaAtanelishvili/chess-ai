"""Checkpoints: model weights plus everything needed to resume training or trust the weights.

A checkpoint records the encoding version it was trained with, and loading refuses a
checkpoint whose version differs from the running code's. Weights trained on one encoding
can never be evaluated with another.

Files are plain ``torch.save`` dictionaries, loaded with ``weights_only=True`` (no arbitrary
code runs when a downloaded checkpoint is opened), and written atomically.
"""

from __future__ import annotations

import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from chessnet import __version__
from chessnet.config import ModelConfig
from chessnet.encoding import ENCODING_VERSION, EncodingVersionError
from chessnet.model.resnet import ChessNet
from chessnet.provenance import git_commit

CHECKPOINT_FORMAT: Final = 1


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """The contents of a checkpoint file."""

    step: int
    epoch: int
    config: dict[str, Any]  # always contains "model": the ModelConfig as a dict
    metrics: dict[str, float]
    encoding_version: int
    chessnet_version: str
    git_sha: str | None
    dataset_manifest_hash: str | None
    created_at: str
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any] | None
    scheduler_state: dict[str, Any] | None
    scaler_state: dict[str, Any] | None
    rng_states: dict[str, Any] | None
    extra: dict[str, Any]  # caller-defined state, e.g. the training loop's progress

    @property
    def model_config(self) -> ModelConfig:
        return ModelConfig.model_validate(self.config["model"])


def capture_rng_states() -> dict[str, Any]:
    """Python, NumPy and torch (CPU and CUDA) RNG states, as weights-only-loadable values."""
    state = np.random.get_state(legacy=True)
    assert isinstance(state, tuple)
    name, keys, position, has_gauss, cached = state
    return {
        "python": random.getstate(),
        "numpy": (
            str(name),
            np.asarray(keys).tolist(),
            int(position),
            int(has_gauss),
            float(cached),
        ),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_states(states: Mapping[str, Any]) -> None:
    """Inverse of :func:`capture_rng_states`."""
    random.setstate(_as_tuple(states["python"]))
    name, keys, position, has_gauss, cached = states["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(states["torch"])
    if states["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


def _as_tuple(value: Any) -> Any:
    """Turn nested lists back into tuples (random.setstate needs tuples)."""
    return tuple(_as_tuple(item) for item in value) if isinstance(value, list | tuple) else value


def save_checkpoint(
    path: str | Path,
    network: ChessNet,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    step: int = 0,
    epoch: int = 0,
    config: Mapping[str, Any] | None = None,
    metrics: Mapping[str, float] | None = None,
    dataset_manifest_hash: str | None = None,
    extra: Mapping[str, Any] | None = None,
    save_rng: bool = True,
) -> Path:
    """Write a checkpoint atomically (a crash mid-write never leaves a broken file).

    ``config`` holds the remaining run configuration (e.g. ``{"train": {...}}``); the model
    configuration is always added under ``"model"``. ``extra`` stores further caller state
    (plain values only), returned as :attr:`Checkpoint.extra`.
    """
    path = Path(path)
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "encoding_version": ENCODING_VERSION,
        "chessnet_version": __version__,
        "git_sha": git_commit(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_manifest_hash": dataset_manifest_hash,
        "config": {**(config or {}), "model": network.config.model_dump()},
        "step": step,
        "epoch": epoch,
        "metrics": dict(metrics or {}),
        "model_state": network.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "rng_states": capture_rng_states() if save_rng else None,
        "extra": dict(extra or {}),
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> Checkpoint:
    """Read a checkpoint.

    Raises:
        EncodingVersionError: if it was trained with a different ``ENCODING_VERSION``.
        ValueError: if it is not a checkpoint of this format.
    """
    payload: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a chessnet checkpoint (format {payload.get('format')})")
    version = payload.get("encoding_version")
    if version != ENCODING_VERSION:
        raise EncodingVersionError(
            f"{path} was trained with encoding version {version}, but this code uses version "
            f"{ENCODING_VERSION}; its weights cannot be used"
        )
    return Checkpoint(
        step=payload["step"],
        epoch=payload["epoch"],
        config=payload["config"],
        metrics=payload["metrics"],
        encoding_version=version,
        chessnet_version=payload["chessnet_version"],
        git_sha=payload["git_sha"],
        dataset_manifest_hash=payload["dataset_manifest_hash"],
        created_at=payload["created_at"],
        model_state=payload["model_state"],
        optimizer_state=payload["optimizer_state"],
        scheduler_state=payload["scheduler_state"],
        scaler_state=payload["scaler_state"],
        rng_states=payload["rng_states"],
        extra=payload.get("extra", {}),
    )


def restore_training_state(
    checkpoint: Checkpoint,
    network: ChessNet,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    restore_rng: bool = False,
) -> None:
    """Load a checkpoint's states into the given objects (each one that is passed)."""
    network.load_state_dict(checkpoint.model_state)
    for target, state in (
        (optimizer, checkpoint.optimizer_state),
        (scheduler, checkpoint.scheduler_state),
        (scaler, checkpoint.scaler_state),
    ):
        if target is not None:
            if state is None:
                raise ValueError("the checkpoint does not contain this state")
            target.load_state_dict(state)
    if restore_rng:
        if checkpoint.rng_states is None:
            raise ValueError("the checkpoint does not contain RNG states")
        restore_rng_states(checkpoint.rng_states)


def load_network(path: str | Path, *, map_location: str | torch.device = "cpu") -> ChessNet:
    """Build a network from a checkpoint's model config and load its weights."""
    checkpoint = load_checkpoint(path, map_location=map_location)
    network = ChessNet(checkpoint.model_config)
    network.load_state_dict(checkpoint.model_state)
    return network
