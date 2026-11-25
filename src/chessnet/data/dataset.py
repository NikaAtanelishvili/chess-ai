"""Training data loading: memory-mapped shards yielding batches of packed records.

Shards are globally shuffled when they are written, so reading consecutive records gives
random batches. Each epoch shuffles the shard order and starts every shard at a random offset.
Batches stay packed (:class:`~chessnet.encoding.packed.PackedBatch`); they are expanded to
input planes on the GPU with :func:`~chessnet.encoding.packed.unpack_batch_torch`.

The batch order of an epoch depends only on the seed, the epoch and the number of DataLoader
workers, so training can resume in the middle of an epoch (:meth:`ShardDataset.start_after`).
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from chessnet.data.shards import Split, load_shard, read_manifest
from chessnet.encoding.packed import PackedBatch, batch_from_records


class ShardDataset(IterableDataset[PackedBatch]):
    """Batches of one split of a dataset directory written by ``chessnet preprocess``.

    With several DataLoader workers, each worker reads a disjoint subset of the shards, and
    the DataLoader takes one batch from each worker in turn. Call :meth:`set_epoch` before
    each epoch to get a new order.

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
        self.shard_lengths = [shard.positions for shard in info.shards]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        # Set by start_after: batches each worker role skips, and which role a fresh
        # DataLoader's first worker takes (so its round-robin continues where it stopped).
        self._skips: list[int] | None = None
        self._phase = 0

    def set_epoch(self, epoch: int) -> None:
        """Select the shuffling of the given epoch (same seed and epoch: same order)."""
        self.epoch = epoch
        self._skips = None
        self._phase = 0

    def _shard_order(self) -> list[int]:
        order = np.arange(len(self.shard_paths))
        if self.shuffle:
            order = np.random.default_rng([self.seed, self.epoch]).permutation(order)
        return [int(index) for index in order]

    def _offset(self, shard_index: int) -> int:
        if not self.shuffle:
            return 0
        return int(
            np.random.default_rng([self.seed, self.epoch, shard_index]).integers(self.batch_size)
        )

    def _batch_count(self, shard_index: int) -> int:
        available = max(self.shard_lengths[shard_index] - self._offset(shard_index), 0)
        if self.drop_last:
            return available // self.batch_size
        return -(-available // self.batch_size)

    def _worker_shards(self, worker_id: int, workers: int) -> list[int]:
        return [
            s for position, s in enumerate(self._shard_order()) if position % workers == worker_id
        ]

    def num_batches(self) -> int:
        """Batches in the current epoch (the same for any number of workers)."""
        return sum(self._batch_count(index) for index in range(len(self.shard_paths)))

    def batches_per_worker(self, num_workers: int) -> list[int]:
        """Batches each DataLoader worker yields this epoch (``num_workers=0``: one stream)."""
        workers = max(num_workers, 1)
        return [
            sum(self._batch_count(s) for s in self._worker_shards(worker, workers))
            for worker in range(workers)
        ]

    def start_after(self, batches: int, *, num_workers: int) -> None:
        """Skip the first ``batches`` batches of the current epoch.

        The skip is computed as a DataLoader with ``num_workers`` workers delivers batches:
        round-robin over the workers, passing over workers that have run dry. A new DataLoader
        starts its round-robin at worker 0, so worker roles are rotated to make it continue
        with the worker that was due next. Skipped batches are never read. Must be called
        after :meth:`set_epoch`, with the DataLoader's worker count.
        """
        remaining = self.batches_per_worker(num_workers)
        if batches > sum(remaining):
            raise ValueError(f"cannot skip {batches} batches; the epoch has {sum(remaining)}")
        skips = [0] * len(remaining)
        worker = 0
        for _ in range(batches):
            while remaining[worker] == 0:
                worker = (worker + 1) % len(remaining)
            skips[worker] += 1
            remaining[worker] -= 1
            worker = (worker + 1) % len(remaining)
        self._skips = skips
        self._phase = worker

    def __iter__(self) -> Iterator[PackedBatch]:
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        role, skip = worker_id, 0
        if self._skips is not None:
            if len(self._skips) != workers:
                raise RuntimeError(
                    f"start_after was computed for {len(self._skips)} workers, "
                    f"but the data is read by {workers}"
                )
            role = (worker_id + self._phase) % workers
            skip = self._skips[role]
        for shard_index in self._worker_shards(role, workers):
            count = self._batch_count(shard_index)
            if skip >= count:
                skip -= count
                continue
            yield from self._batches(shard_index, first=skip)
            skip = 0

    def _batches(self, shard_index: int, *, first: int = 0) -> Iterator[PackedBatch]:
        records = load_shard(self.shard_paths[shard_index])
        start = self._offset(shard_index) + first * self.batch_size
        for begin in range(start, len(records), self.batch_size):
            stop = begin + self.batch_size
            if stop > len(records) and self.drop_last:
                return
            yield batch_from_records(records[begin:stop])


def make_loader(
    dataset: ShardDataset, *, num_workers: int = 2, prefetch_factor: int = 4
) -> DataLoader[PackedBatch]:
    """A DataLoader that passes the dataset's batches through unchanged.

    Workers start through ``forkserver`` where available: forking the training process, which
    runs threads (TensorBoard, torch), can deadlock the children.
    """
    context = None
    if num_workers > 0 and "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        multiprocessing_context=context,
    )
