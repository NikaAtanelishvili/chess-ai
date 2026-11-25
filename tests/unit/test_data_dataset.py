import json
import shutil
from pathlib import Path

import pytest
import torch

from chessnet.config import DataConfig
from chessnet.data.dataset import ShardDataset, make_loader
from chessnet.data.pipeline import preprocess
from chessnet.data.shards import MANIFEST_NAME
from chessnet.encoding import EncodingVersionError
from chessnet.encoding.packed import PackedBatch, unpack_batch_torch

FIXTURES = Path(__file__).parents[1] / "fixtures"
TRAIN_POSITIONS = 942  # fixture games 01-08, 11 and 18


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("dataset") / "shards"
    config = DataConfig(shard_size=100, num_buckets=3, seed=5)
    preprocess(config, str(FIXTURES / "mini.pgn.zst"), out, workers=0)
    return out


def tags(batches: list[PackedBatch]) -> list[tuple[int, int, int]]:
    """Identify records by (ply, move, elo) for comparisons."""
    return [
        (int(p), int(m), int(e))
        for batch in batches
        for p, m, e in zip(batch.ply, batch.move, batch.elo, strict=True)
    ]


def test_unshuffled_pass_covers_the_split(dataset_dir: Path) -> None:
    dataset = ShardDataset(dataset_dir, "train", batch_size=64, shuffle=False, drop_last=False)
    assert dataset.positions == TRAIN_POSITIONS
    assert len(dataset.shard_paths) == 10  # 100 positions per shard
    batches = list(dataset)
    assert sum(len(batch) for batch in batches) == TRAIN_POSITIONS
    assert all(batch.bitboards.shape == (len(batch), 12) for batch in batches)


def test_drop_last_yields_full_batches_only(dataset_dir: Path) -> None:
    batches = list(ShardDataset(dataset_dir, batch_size=64, shuffle=False))
    assert batches
    assert {len(batch) for batch in batches} == {64}


def test_shuffling_depends_on_seed_and_epoch(dataset_dir: Path) -> None:
    dataset = ShardDataset(dataset_dir, batch_size=32, seed=3)
    epoch0 = tags(list(dataset))
    assert tags(list(dataset)) == epoch0  # same epoch, same order
    dataset.set_epoch(1)
    assert tags(list(dataset)) != epoch0
    other_seed = ShardDataset(dataset_dir, batch_size=32, seed=4)
    assert tags(list(other_seed)) != epoch0


def test_workers_read_disjoint_shards(dataset_dir: Path) -> None:
    dataset = ShardDataset(dataset_dir, batch_size=50, shuffle=False, drop_last=False)
    single = sorted(tags(list(dataset)))
    loaded = sorted(tags(list(make_loader(dataset, num_workers=2, prefetch_factor=2))))
    assert loaded == single
    assert len(loaded) == TRAIN_POSITIONS


def test_loader_batches_unpack_to_planes(dataset_dir: Path) -> None:
    dataset = ShardDataset(dataset_dir, "val", batch_size=16)
    batch = next(iter(make_loader(dataset, num_workers=0)))
    planes = unpack_batch_torch(batch)
    assert planes.shape == (16, 19, 8, 8)
    assert planes.dtype == torch.float32
    assert batch.move.dtype == torch.int64
    assert int(batch.move.min()) >= 0
    assert int(batch.move.max()) < 4672
    assert set(batch.wdl.tolist()) <= {0, 1, 2}


def test_invalid_batch_size(dataset_dir: Path) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        ShardDataset(dataset_dir, batch_size=0)


def test_other_encoding_versions_are_refused(dataset_dir: Path, tmp_path: Path) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(dataset_dir, copy)
    manifest = json.loads((copy / MANIFEST_NAME).read_text())
    manifest["encoding_version"] = 0
    (copy / MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(EncodingVersionError):
        ShardDataset(copy, batch_size=8)


def batch_keys(batches: list[PackedBatch]) -> list[tuple[tuple[int, int, int], ...]]:
    """Each batch as the tuple of its records' (ply, move, elo)."""
    return [
        tuple(zip(b.ply.tolist(), b.move.tolist(), b.elo.tolist(), strict=True)) for b in batches
    ]


def test_batch_counts(dataset_dir: Path) -> None:
    dataset = ShardDataset(dataset_dir, batch_size=16, seed=3)
    dataset.set_epoch(1)
    assert dataset.num_batches() == len(list(dataset))
    for workers in (0, 1, 2, 3):
        counts = dataset.batches_per_worker(workers)
        assert len(counts) == max(workers, 1)
        assert sum(counts) == dataset.num_batches()


@pytest.mark.parametrize("num_workers", [0, 1, 2, 3])
def test_start_after_resumes_where_a_loader_left_off(dataset_dir: Path, num_workers: int) -> None:
    dataset = ShardDataset(dataset_dir, batch_size=16, seed=3)
    dataset.set_epoch(2)
    full = batch_keys(list(make_loader(dataset, num_workers=num_workers, prefetch_factor=2)))
    assert len(full) == dataset.num_batches()
    for skip in (0, 1, 7, len(full) - 3, len(full)):
        dataset.set_epoch(2)
        dataset.start_after(skip, num_workers=num_workers)
        resumed = list(make_loader(dataset, num_workers=num_workers, prefetch_factor=2))
        assert batch_keys(resumed) == full[skip:], skip


def test_start_after_checks_its_arguments(dataset_dir: Path) -> None:
    dataset = ShardDataset(dataset_dir, batch_size=16, seed=3)
    with pytest.raises(ValueError, match="cannot skip"):
        dataset.start_after(dataset.num_batches() + 1, num_workers=0)
    dataset.start_after(2, num_workers=2)
    with pytest.raises(RuntimeError, match="computed for 2 workers"):
        list(dataset)  # read by one process, not two workers
    dataset.set_epoch(0)  # a new epoch clears the skip
    assert len(list(dataset)) == dataset.num_batches()
