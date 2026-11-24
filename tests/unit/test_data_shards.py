import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from chessnet.config import DataConfig
from chessnet.data.shards import (
    MANIFEST_NAME,
    RECORD_DTYPE_DESCRIPTION,
    GameCounts,
    Manifest,
    ShardWriter,
    SourceInfo,
    Split,
    SplitInfo,
    load_shard,
    manifest_hash,
    read_manifest,
    split_of_site,
    write_manifest,
)
from chessnet.encoding import ENCODING_VERSION, EncodingVersionError
from chessnet.encoding.packed import RECORD_SIZE, empty_records

# Fixture games 09 and 10 were given URLs that hash into the val and test splits.
VAL_SITE = "https://lichess.org/game/fixt09c9"
TEST_SITE = "https://lichess.org/game/fixt10dV"


def tagged_records(count: int, first_tag: int = 0) -> np.ndarray:
    """Records whose ``elo`` field holds a unique tag (the content is irrelevant here)."""
    records = empty_records(count)
    records["elo"] = np.arange(first_tag, first_tag + count)
    return records


def test_split_is_stable() -> None:
    assert split_of_site(VAL_SITE) == Split.VAL
    assert split_of_site(TEST_SITE) == Split.TEST
    assert split_of_site("https://lichess.org/game/fixt01aa") == Split.TRAIN
    assert split_of_site(VAL_SITE) == split_of_site(VAL_SITE)


def test_split_fractions() -> None:
    rng = random.Random(0)
    sites = [f"https://lichess.org/game/{rng.getrandbits(48):012x}" for _ in range(50_000)]
    counts = Counter(split_of_site(site) for site in sites)
    assert counts[Split.TRAIN] / len(sites) == pytest.approx(0.98, abs=0.005)
    assert counts[Split.VAL] / len(sites) == pytest.approx(0.01, abs=0.002)
    assert counts[Split.TEST] / len(sites) == pytest.approx(0.01, abs=0.002)


def write_games(directory: Path, games: list[tuple[Split, int]], **options: int) -> ShardWriter:
    writer = ShardWriter(directory, **({"num_buckets": 5, "shard_size": 100, "seed": 1} | options))
    tag = 0
    for split, size in games:
        writer.add(split, tagged_records(size, tag))
        tag += size
    return writer


# The one-position game leaves most buckets empty in its flush (when flushing per game).
GAMES = [(Split.TRAIN, 70)] * 20 + [
    (Split.VAL, 33),
    (Split.TEST, 41),
    (Split.TRAIN, 9),
    (Split.TRAIN, 1),
]


@pytest.mark.parametrize("flush_records", [1, 64, 1 << 17])
def test_shards_hold_every_record_once(tmp_path: Path, flush_records: int) -> None:
    writer = write_games(tmp_path, GAMES, flush_records=flush_records)
    shards = writer.finalize()
    assert not (tmp_path / "buckets").exists()

    expected: dict[Split, list[int]] = {split: [] for split in Split}
    tag = 0
    for game_split, size in GAMES:
        expected[game_split] += range(tag, tag + size)
        tag += size
    for split in Split:
        seen: list[int] = []
        for info in shards[split]:
            records = load_shard(tmp_path / info.file)
            assert len(records) == info.positions
            seen += records["elo"].tolist()
            digest = hashlib.sha256((tmp_path / info.file).read_bytes()).hexdigest()
            assert digest == info.sha256
        assert sorted(seen) == expected[split], split
        assert writer.positions[split] == len(expected[split])


def test_every_shard_but_the_last_is_full(tmp_path: Path) -> None:
    shards = write_games(tmp_path, GAMES).finalize()
    sizes = [info.positions for info in shards[Split.TRAIN]]
    assert sum(sizes) == 70 * 20 + 9 + 1
    assert sizes[:-1] == [100] * (len(sizes) - 1)
    assert 0 < sizes[-1] <= 100
    assert [info.file for info in shards[Split.TRAIN]][:2] == ["train-00000.npy", "train-00001.npy"]


def test_positions_of_a_game_are_spread_out(tmp_path: Path) -> None:
    shards = write_games(tmp_path, GAMES).finalize()
    order = np.concatenate(
        [load_shard(tmp_path / info.file)["elo"] for info in shards[Split.TRAIN]]
    )
    first_game = np.flatnonzero(order < 70)  # where the 70 positions of game 0 ended up
    assert first_game.max() - first_game.min() > 700


def shard_bytes(directory: Path, shards: dict[Split, list]) -> list[bytes]:
    return [(directory / info.file).read_bytes() for split in Split for info in shards[split]]


def test_output_is_determined_by_the_seed(tmp_path: Path) -> None:
    first = write_games(tmp_path / "a", GAMES).finalize()
    second = write_games(tmp_path / "b", GAMES).finalize()
    other = write_games(tmp_path / "c", GAMES, seed=2).finalize()
    assert shard_bytes(tmp_path / "a", first) == shard_bytes(tmp_path / "b", second)
    assert shard_bytes(tmp_path / "a", first) != shard_bytes(tmp_path / "c", other)


def test_an_empty_split_has_no_shards(tmp_path: Path) -> None:
    shards = write_games(tmp_path, [(Split.TRAIN, 10)]).finalize()
    assert shards[Split.VAL] == []
    assert shards[Split.TEST] == []


def make_manifest(**overrides: object) -> Manifest:
    values: dict[str, object] = {
        "encoding_version": ENCODING_VERSION,
        "record_size": RECORD_SIZE,
        "record_dtype": RECORD_DTYPE_DESCRIPTION,
        "chessnet_version": "0.0.0",
        "git_commit": None,
        "created_at": "2026-11-25T00:00:00+00:00",
        "elapsed_seconds": 1.0,
        "config": DataConfig(),
        "assumptions": ("test assumption",),
        "source": SourceInfo(
            location="x.pgn",
            kind="path",
            compressed=False,
            bytes_read=1,
            bytes_decompressed=1,
            stopped_because="end_of_source",
        ),
        "games": GameCounts(scanned=1, passed_header_filter=1, kept=1, rejected={}),
        "skipped_positions": 0,
        "splits": {split: SplitInfo(games=0, positions=0, shards=()) for split in Split},
    }
    return Manifest.model_validate(values | overrides)


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = make_manifest()
    path = write_manifest(tmp_path, manifest)
    assert path == tmp_path / MANIFEST_NAME
    assert not list(tmp_path.glob("*.tmp"))
    assert read_manifest(tmp_path) == manifest
    assert manifest_hash(tmp_path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_data_from_another_encoding_version_is_refused(tmp_path: Path) -> None:
    write_manifest(tmp_path, make_manifest())
    raw = json.loads((tmp_path / MANIFEST_NAME).read_text())
    raw["encoding_version"] = ENCODING_VERSION + 1
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(raw))
    with pytest.raises(EncodingVersionError, match="regenerate"):
        read_manifest(tmp_path)


def test_shards_of_another_format_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "train-00000.npy"
    np.save(path, np.zeros(10, dtype=np.float32))
    with pytest.raises(EncodingVersionError):
        load_shard(path)
