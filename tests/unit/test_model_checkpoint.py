import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from chessnet import __version__
from chessnet.config import MODEL_PRESETS
from chessnet.encoding import ENCODING_VERSION, EncodingVersionError
from chessnet.model.checkpoint import (
    CHECKPOINT_FORMAT,
    load_checkpoint,
    load_network,
    restore_training_state,
    save_checkpoint,
)
from chessnet.model.resnet import ChessNet

TINY = MODEL_PRESETS["tiny"]


def training_objects() -> tuple[
    ChessNet,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    torch.amp.GradScaler,
]:
    network = ChessNet(TINY)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
    return network, optimizer, scheduler, scaler


def train_one_step(network: ChessNet, optimizer: torch.optim.Optimizer) -> None:
    output = network(torch.randn(4, 19, 8, 8))
    loss = F.cross_entropy(output.policy, torch.tensor([0, 1, 2, 3])) + output.value.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def assert_states_equal(first: Any, second: Any) -> None:
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_states_equal(first[key], second[key])
    elif isinstance(first, list | tuple):
        assert len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            assert_states_equal(a, b)
    else:
        assert first == second


def test_round_trip_restores_everything(tmp_path: Path) -> None:
    torch.manual_seed(0)
    network, optimizer, scheduler, scaler = training_objects()
    train_one_step(network, optimizer)
    scheduler.step()
    path = save_checkpoint(
        tmp_path / "step-42.pt",
        network,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        step=42,
        epoch=3,
        config={"train": {"lr": 1e-3, "batch_size": 4}},
        metrics={"val_top1": 0.25},
        dataset_manifest_hash="ab" * 32,
    )
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))

    checkpoint = load_checkpoint(path)
    assert (checkpoint.step, checkpoint.epoch) == (42, 3)
    assert checkpoint.metrics == {"val_top1": 0.25}
    assert checkpoint.config["train"] == {"lr": 1e-3, "batch_size": 4}
    assert checkpoint.model_config == TINY
    assert checkpoint.encoding_version == ENCODING_VERSION
    assert checkpoint.chessnet_version == __version__
    assert checkpoint.dataset_manifest_hash == "ab" * 32
    assert checkpoint.git_sha is not None
    assert re.fullmatch(r"[0-9a-f]{40}(-dirty)?", checkpoint.git_sha)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", checkpoint.created_at)

    restored = training_objects()
    restore_training_state(
        checkpoint,
        restored[0],
        optimizer=restored[1],
        scheduler=restored[2],
        scaler=restored[3],
    )
    for original, copy in zip((network, optimizer, scheduler, scaler), restored, strict=True):
        assert_states_equal(original.state_dict(), copy.state_dict())


def test_rng_states_are_restored(tmp_path: Path) -> None:
    network = ChessNet(TINY)
    path = save_checkpoint(tmp_path / "rng.pt", network)
    expected = (random.random(), float(np.random.random()), float(torch.rand(1)))
    random.random(), np.random.random(), torch.rand(1)  # move the generators on
    restore_training_state(load_checkpoint(path), network, restore_rng=True)
    assert (random.random(), float(np.random.random()), float(torch.rand(1))) == expected


def test_load_network_rebuilds_the_model(tmp_path: Path) -> None:
    network = ChessNet(TINY).eval()
    path = save_checkpoint(tmp_path / "model.pt", network, save_rng=False)
    rebuilt = load_network(path).eval()
    planes = torch.randn(3, 19, 8, 8)
    with torch.no_grad():
        first, second = network(planes), rebuilt(planes)
    assert torch.equal(first.policy, second.policy)
    assert torch.equal(first.value, second.value)


def rewrite(path: Path, **changes: Any) -> None:
    payload = torch.load(path, weights_only=True)
    payload.update(changes)
    torch.save(payload, path)


def test_checkpoints_from_another_encoding_version_are_refused(tmp_path: Path) -> None:
    path = save_checkpoint(tmp_path / "old.pt", ChessNet(TINY))
    rewrite(path, encoding_version=ENCODING_VERSION + 1)
    with pytest.raises(EncodingVersionError, match="cannot be used"):
        load_checkpoint(path)
    with pytest.raises(EncodingVersionError):
        load_network(path)


def test_files_of_another_format_are_refused(tmp_path: Path) -> None:
    path = save_checkpoint(tmp_path / "odd.pt", ChessNet(TINY))
    rewrite(path, format=CHECKPOINT_FORMAT + 1)
    with pytest.raises(ValueError, match="not a chessnet checkpoint"):
        load_checkpoint(path)
    plain = tmp_path / "weights.pt"
    torch.save(ChessNet(TINY).state_dict(), plain)
    with pytest.raises(ValueError, match="not a chessnet checkpoint"):
        load_checkpoint(plain)


def test_missing_states_cannot_be_restored(tmp_path: Path) -> None:
    network, optimizer, _, _ = training_objects()
    path = save_checkpoint(tmp_path / "weights-only.pt", network, save_rng=False)
    checkpoint = load_checkpoint(path)
    assert checkpoint.optimizer_state is None
    assert checkpoint.rng_states is None
    with pytest.raises(ValueError, match="does not contain"):
        restore_training_state(checkpoint, network, optimizer=optimizer)
    with pytest.raises(ValueError, match="RNG"):
        restore_training_state(checkpoint, network, restore_rng=True)
