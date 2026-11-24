"""Training data loading: memory-mapped shards yielding batches of packed records.

Shards are globally shuffled when they are written, so reading consecutive records gives
random batches. Each epoch shuffles the shard order and starts every shard at a random offset.
Batches stay packed (:class:`~chessnet.encoding.packed.PackedBatch`); they are expanded to
input planes on the GPU with :func:`~chessnet.encoding.packed.unpack_batch_torch`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from chessnet.data.shards import Split, load_shard, read_manifest
from chessnet.encoding.packed import PackedBatch, batch_from_records


class ShardDataset(IterableDataset[PackedBatch]):
    """Batches of one split of a dataset directory written by ``chessnet preprocess``.

    With several DataLoader workers, each worker reads a disjoint subset of the shards.
    Call :meth:`set_epoch` before each epoch to get a new order.

    Raises:
        EncodingVersionError: if the dataset was built with another encoding version.
    """

    def __init__(
        self,
        directory: str | Path,
        split: Split | str = Split.TRAIN,
        *,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.directory = Path(directory)
        self.split = Split(split)
        manifest = read_manifest(self.directory)
        info = manifest.splits[self.split]
        self.positions = info.positions
        self.shard_paths = [self.directory / shard.file for shard in info.shards]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select the shuffling of the given epoch (same seed and epoch: same order)."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[PackedBatch]:
        worker = get_worker_info()
        worker_id, num_workers = (worker.id, worker.num_workers) if worker else (0, 1)
        order = np.arange(len(self.shard_paths))
        if self.shuffle:
            order = np.random.default_rng([self.seed, self.epoch]).permutation(order)
        for position, shard_index in enumerate(order.tolist()):
            if position % num_workers == worker_id:
                yield from self._batches(shard_index)

    def _batches(self, shard_index: int) -> Iterator[PackedBatch]:
        records = load_shard(self.shard_paths[shard_index])
        offset = 0
        if self.shuffle:
            rng = np.random.default_rng([self.seed, self.epoch, shard_index])
            offset = int(rng.integers(self.batch_size))
        for start in range(offset, len(records), self.batch_size):
            stop = start + self.batch_size
            if stop > len(records) and self.drop_last:
                return
            yield batch_from_records(records[start:stop])


def make_loader(
    dataset: ShardDataset, *, num_workers: int = 2, prefetch_factor: int = 4
) -> DataLoader[PackedBatch]:
    """A DataLoader that passes the dataset's batches through unchanged."""
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
