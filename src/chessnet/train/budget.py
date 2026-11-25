"""Run-time estimates for fitting a training run into a wall-time budget.

An estimate adds up what a run spends its time on: training steps at a measured speed,
validation passes (each possibly followed by a ``best.pt`` write), rolling and end-of-epoch
checkpoints, a fresh DataLoader each epoch, and start-up, including the first, compiling
step. Pure Python, so the Kaggle kernel can use it without importing torch.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class RunCosts:
    """Measured speeds and fixed costs of a training run. Times are in seconds."""

    steps_per_epoch: int
    batch_size: int
    train_speed: float  # training positions per second
    eval_speed: float  # forward-only positions per second, as in validation
    val_every: int  # steps between validations
    val_positions: int  # positions per validation
    checkpoint_minutes: float  # wall-clock interval of rolling checkpoints
    checkpoint_seconds: float = 1.0  # one checkpoint write
    startup_seconds: float = 120.0  # imports, workers, validation subset, compilation
    epoch_restart_seconds: float = 10.0  # new DataLoader workers each epoch

    def __post_init__(self) -> None:
        counts = (self.steps_per_epoch, self.batch_size, self.val_every, self.val_positions)
        rates = (self.train_speed, self.eval_speed, self.checkpoint_minutes)
        if min(counts) < 1 or not all(rate > 0 for rate in rates):  # NaN fails too
            raise ValueError(f"run costs need positive counts and speeds: {self}")


def steps_per_epoch(shard_positions: Iterable[int], batch_size: int) -> int:
    """Training steps in one epoch, at most: each shard drops its last partial batch.

    Shuffled epochs start each shard at a random offset below one batch, so an epoch can
    be up to one step per shard shorter.
    """
    return sum(positions // batch_size for positions in shard_positions)


def estimate_seconds(epochs: int, costs: RunCosts) -> float:
    """Estimated wall time of a fresh run of ``epochs`` epochs."""
    steps = epochs * costs.steps_per_epoch
    training = steps * costs.batch_size / costs.train_speed
    validations = steps // costs.val_every + 1  # the final step validates too
    validating = validations * (costs.val_positions / costs.eval_speed + costs.checkpoint_seconds)
    checkpoints = math.ceil(training / (costs.checkpoint_minutes * 60)) + epochs + 1
    return (
        training
        + validating
        + checkpoints * costs.checkpoint_seconds
        + costs.startup_seconds
        + epochs * costs.epoch_restart_seconds
    )


def max_epochs(budget_seconds: float, margin_seconds: float, costs: RunCosts) -> int:
    """The most whole epochs whose estimate leaves ``margin_seconds`` of the budget; 0 if none."""
    available = budget_seconds - margin_seconds
    epoch_training = costs.steps_per_epoch * costs.batch_size / costs.train_speed
    low, high = 0, max(int(available / epoch_training), 0)  # training time alone bounds it
    while low < high:  # estimates grow with the epochs: binary search for the last that fits
        middle = (low + high + 1) // 2
        if estimate_seconds(middle, costs) <= available:
            low = middle
        else:
            high = middle - 1
    return low
