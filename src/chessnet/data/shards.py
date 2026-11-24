"""Dataset splits, shuffled shard files and the dataset manifest.

Splitting is by game: a hash of the game's URL decides train, val or test, so positions of
one game never leak across splits. Shuffling is global and takes two passes over the data:

1. While games stream in, every position goes to a random bucket file on disk.
2. Each bucket is loaded, permuted and written out as final shards (``train-00000.npy``,
   ...), and deleted right away. Peak disk use stays near one dataset plus one bucket.

Positions of a game are spread over all buckets, so each shard is a random sample of the
whole split.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from chessnet.config import DataConfig
from chessnet.encoding import ENCODING_VERSION
from chessnet.encoding.packed import PACKED_DTYPE, PackedRecords

MANIFEST_NAME = "manifest.json"
#: The record layout as stored in manifests (JSON-normalised: lists, not tuples).
RECORD_DTYPE_DESCRIPTION: list[Any] = json.loads(json.dumps(PACKED_DTYPE.descr))


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


#: Of 100 hash buckets, 0-97 are train, 98 val and 99 test.
SPLIT_BUCKETS = {Split.TRAIN: 98, Split.VAL: 1, Split.TEST: 1}


def split_of_site(site: str) -> Split:
    """Split of a game, from a stable hash (BLAKE2b) of its URL (the PGN ``Site`` tag)."""
    digest = hashlib.blake2b(site.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "little") % 100
    if bucket < SPLIT_BUCKETS[Split.TRAIN]:
        return Split.TRAIN
    return Split.VAL if bucket == SPLIT_BUCKETS[Split.TRAIN] else Split.TEST


class EncodingVersionError(ValueError):
    """Data was written with a different ``ENCODING_VERSION`` than the running code."""


class ShardInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str
    positions: int
    sha256: str


class SplitInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    games: int
    positions: int
    shards: tuple[ShardInfo, ...]


class SourceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    location: str  # URL or path
    kind: str  # "url" or "path"
    compressed: bool
    bytes_read: int  # from the network or file, including read-ahead
    bytes_decompressed: int
    stopped_because: str  # "max_games" or "end_of_source"


class GameCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scanned: int
    passed_header_filter: int
    kept: int
    rejected: dict[str, int]


class Manifest(BaseModel):
    """Everything needed to trust, reproduce and load a dataset directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    encoding_version: int
    record_size: int
    record_dtype: list[Any]
    chessnet_version: str
    git_commit: str | None
    created_at: str  # ISO 8601, UTC
    elapsed_seconds: float
    config: DataConfig
    source: SourceInfo
    games: GameCounts
    skipped_positions: int
    splits: dict[Split, SplitInfo]


def write_manifest(directory: Path, manifest: Manifest) -> Path:
    """Write ``manifest.json`` atomically (written last, it marks the dataset as complete)."""
    path = directory / MANIFEST_NAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def read_manifest(directory: Path) -> Manifest:
    """Read a dataset's manifest, refusing data built with another encoding version."""
    raw = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    version = raw.get("encoding_version")
    if version != ENCODING_VERSION:
        raise EncodingVersionError(
            f"{directory} was built with encoding version {version}, but this code uses "
            f"version {ENCODING_VERSION}; regenerate the dataset"
        )
    return Manifest.model_validate(raw)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ShardWriter:
    """Collect packed records per split into random buckets, then write shuffled shards."""

    def __init__(
        self,
        directory: Path,
        *,
        num_buckets: int,
        shard_size: int,
        seed: int,
        flush_records: int = 1 << 17,
    ) -> None:
        self._directory = directory
        self._bucket_directory = directory / "buckets"
        self._bucket_directory.mkdir(parents=True, exist_ok=True)
        self._shard_size = shard_size
        self._seed = seed
        self._flush_records = flush_records
        train_share = SPLIT_BUCKETS[Split.TRAIN]
        self._num_buckets = {
            split: max(1, round(num_buckets * share / train_share))
            for split, share in SPLIT_BUCKETS.items()
        }
        self._assign_rngs = {
            split: np.random.default_rng([seed, 0, index]) for index, split in enumerate(Split)
        }
        self._pending: dict[Split, list[PackedRecords]] = {split: [] for split in Split}
        self._pending_records = dict.fromkeys(Split, 0)
        self.positions = dict.fromkeys(Split, 0)

    def _bucket_path(self, split: Split, bucket: int) -> Path:
        return self._bucket_directory / f"{split}-{bucket:04d}.bin"

    def add(self, split: Split, records: PackedRecords) -> None:
        """Queue one game's records; they reach the bucket files in batches."""
        self._pending[split].append(records)
        self._pending_records[split] += len(records)
        self.positions[split] += len(records)
        if self._pending_records[split] >= self._flush_records:
            self._flush(split)

    def _flush(self, split: Split) -> None:
        if not self._pending[split]:
            return
        records = np.concatenate(self._pending[split])
        self._pending[split] = []
        self._pending_records[split] = 0
        count = self._num_buckets[split]
        buckets = self._assign_rngs[split].integers(0, count, size=len(records))
        records = records[np.argsort(buckets, kind="stable")]
        ends = np.cumsum(np.bincount(buckets, minlength=count))
        start = 0
        for bucket, end in enumerate(ends.tolist()):
            if end > start:
                with self._bucket_path(split, bucket).open("ab") as file:
                    records[start:end].tofile(file)
            start = end

    def _buckets(self, split: Split) -> Iterator[Path]:
        for bucket in range(self._num_buckets[split]):
            path = self._bucket_path(split, bucket)
            if path.exists():
                yield path

    def finalize(self) -> dict[Split, list[ShardInfo]]:
        """Second pass: permute each bucket, delete it, and cut the stream into shards.

        Every shard holds exactly ``shard_size`` positions except the last of each split.
        """
        shards: dict[Split, list[ShardInfo]] = {split: [] for split in Split}
        for index, split in enumerate(Split):
            self._flush(split)
            permute_rng = np.random.default_rng([self._seed, 1, index])
            pending: list[PackedRecords] = []
            pending_count = 0
            for bucket_path in self._buckets(split):
                records = np.fromfile(bucket_path, dtype=PACKED_DTYPE)
                bucket_path.unlink()
                pending.append(records[permute_rng.permutation(len(records))])
                pending_count += len(records)
                while pending_count >= self._shard_size:
                    stream = np.concatenate(pending)
                    shards[split].append(
                        self._write_shard(split, len(shards[split]), stream[: self._shard_size])
                    )
                    pending = [stream[self._shard_size :]]
                    pending_count -= self._shard_size
            if pending_count:
                shards[split].append(
                    self._write_shard(split, len(shards[split]), np.concatenate(pending))
                )
        self._bucket_directory.rmdir()
        return shards

    def _write_shard(self, split: Split, number: int, records: PackedRecords) -> ShardInfo:
        name = f"{split}-{number:05d}.npy"
        path = self._directory / name
        np.save(path, records)
        return ShardInfo(file=name, positions=len(records), sha256=_sha256(path))


def load_shard(path: Path) -> PackedRecords:
    """Memory-map a shard written by :class:`ShardWriter`."""
    records: PackedRecords = np.load(path, mmap_mode="r")
    if records.dtype != PACKED_DTYPE:
        raise EncodingVersionError(f"{path} does not contain packed records of this version")
    return records
