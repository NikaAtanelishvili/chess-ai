"""Small synthetic datasets (random legal games) for training tests."""

import random
from pathlib import Path

import chess

from chessnet import __version__
from chessnet.config import DataConfig
from chessnet.data.shards import (
    RECORD_DTYPE_DESCRIPTION,
    GameCounts,
    Manifest,
    ShardWriter,
    SourceInfo,
    Split,
    SplitInfo,
    write_manifest,
)
from chessnet.encoding import ENCODING_VERSION
from chessnet.encoding.packed import RECORD_SIZE, empty_records, pack_into
from chessnet.encoding.value import wdl_from_result

from helpers.positions import PLAYOUT_STARTS


def make_dataset(
    directory: Path, *, positions: int = 2400, seed: int = 0, shard_size: int = 500
) -> Path:
    """Write about ``positions`` positions of random games as a dataset (10% of games: val)."""
    rng = random.Random(seed)
    writer = ShardWriter(directory, num_buckets=4, shard_size=shard_size, seed=seed)
    games = dict.fromkeys(Split, 0)
    written = 0
    game_number = 0
    while written < positions:
        board = chess.Board(PLAYOUT_STARTS[game_number % len(PLAYOUT_STARTS)])
        result = rng.choice(["1-0", "0-1", "1/2-1/2"])
        examples = []
        while not board.is_game_over() and len(examples) < 80:
            move = rng.choice(list(board.legal_moves))
            examples.append((board.copy(stack=False), move))
            board.push(move)
        records = empty_records(len(examples))
        for index, (position, move) in enumerate(examples):
            pack_into(
                records,
                index,
                position,
                move,
                wdl_from_result(result, position.turn),
                elo=rng.randint(2000, 2600),
            )
        split = Split.VAL if game_number % 10 == 9 else Split.TRAIN
        writer.add(split, records)
        games[split] += 1
        written += len(records)
        game_number += 1
    shards = writer.finalize()
    write_manifest(
        directory,
        Manifest(
            encoding_version=ENCODING_VERSION,
            record_size=RECORD_SIZE,
            record_dtype=RECORD_DTYPE_DESCRIPTION,
            chessnet_version=__version__,
            git_commit=None,
            created_at="2026-11-25T00:00:00+00:00",
            elapsed_seconds=0.0,
            config=DataConfig(shard_size=shard_size, num_buckets=4, seed=seed),
            assumptions=("synthetic test data: random legal games",),
            source=SourceInfo(
                location=f"synthetic:seed={seed}",
                kind="path",
                compressed=False,
                bytes_read=0,
                bytes_decompressed=0,
                stopped_because="end_of_source",
            ),
            games=GameCounts(
                scanned=game_number, passed_header_filter=game_number, kept=game_number, rejected={}
            ),
            skipped_positions=0,
            splits={
                split: SplitInfo(
                    games=games[split],
                    positions=writer.positions[split],
                    shards=tuple(shards[split]),
                )
                for split in Split
            },
        ),
    )
    return directory
