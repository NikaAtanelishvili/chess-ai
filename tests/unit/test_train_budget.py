"""Run-time estimates: the main run's epoch count, and the rule the second run is sized by."""

from pathlib import Path

import pytest

from chessnet.data.dataset import ShardDataset
from chessnet.data.shards import Split, read_manifest
from chessnet.train.budget import RunCosts, estimate_seconds, max_epochs, steps_per_epoch

from helpers.datasets import make_dataset

WALL_TIME = 11 * 3600 + 15 * 60

# The main run as measured in session 1: base, batch 2048, compiled, on one T4.
MAIN = RunCosts(
    steps_per_epoch=37_572,
    batch_size=2048,
    train_speed=10_639,
    eval_speed=20_514,
    val_every=2500,
    val_positions=25 * 2048,
    checkpoint_minutes=20,
)


def test_the_main_run_fits_five_epochs_with_an_hour_to_spare() -> None:
    assert max_epochs(WALL_TIME, 3600, MAIN) == 5
    assert estimate_seconds(5, MAIN) / 3600 == pytest.approx(10.18, abs=0.005)  # TRAINING.md
    assert estimate_seconds(6, MAIN) + 3600 > WALL_TIME


def test_overheads_come_on_top_of_training() -> None:
    training = MAIN.steps_per_epoch * MAIN.batch_size / MAIN.train_speed
    overhead = estimate_seconds(1, MAIN) - training
    assert MAIN.startup_seconds < overhead < 10 * 60
    assert estimate_seconds(2, MAIN) > estimate_seconds(1, MAIN) + training


@pytest.mark.parametrize("hours", [2.5, 4.0, 9.0, 11.25, 1000.0])
@pytest.mark.parametrize("margin", [0.0, 3600.0, 5400.0])
def test_max_epochs_is_the_most_that_leaves_the_margin(hours: float, margin: float) -> None:
    budget = hours * 3600
    epochs = max_epochs(budget, margin, MAIN)
    if epochs:
        assert estimate_seconds(epochs, MAIN) + margin <= budget
    assert estimate_seconds(epochs + 1, MAIN) + margin > budget


def test_nothing_fits_when_one_epoch_is_too_long() -> None:
    assert max_epochs(3 * 3600, 5400, MAIN) == 0  # one epoch alone takes over 2 h
    assert max_epochs(1000, 5400, MAIN) == 0  # less time than the margin


def test_steps_per_epoch_drops_each_shards_partial_batch() -> None:
    assert steps_per_epoch([5000, 4096, 100], 2048) == 2 + 2 + 0


def test_steps_per_epoch_bounds_what_the_dataset_yields(tmp_path: Path) -> None:
    data = make_dataset(tmp_path / "shards", positions=2400, shard_size=500)
    sizes = [shard.positions for shard in read_manifest(data).splits[Split.TRAIN].shards]
    expected = steps_per_epoch(sizes, 64)
    assert ShardDataset(data, Split.TRAIN, batch_size=64, shuffle=False).num_batches() == expected
    shuffled = ShardDataset(data, Split.TRAIN, batch_size=64, shuffle=True, seed=0)
    assert expected - len(sizes) <= shuffled.num_batches() <= expected


@pytest.mark.parametrize(
    "change", [{"train_speed": 0.0}, {"eval_speed": float("nan")}, {"steps_per_epoch": 0}]
)
def test_impossible_costs_are_refused(change: dict[str, float]) -> None:
    fields = {**MAIN.__dict__, **change}
    with pytest.raises(ValueError, match="positive counts and speeds"):
        RunCosts(**fields)
