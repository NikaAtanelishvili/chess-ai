"""End to end: fixture PGN dump -> shards + manifest -> data loader.

The expected numbers come from the hand-checked table in tests/fixtures/README.md.
"""

import hashlib
import io
import re
from collections import Counter
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from chessnet.cli import app
from chessnet.config import DataConfig
from chessnet.data import pipeline
from chessnet.data.dataset import ShardDataset, make_loader
from chessnet.data.pipeline import PreprocessError, preprocess
from chessnet.data.shards import MANIFEST_NAME, Manifest, Split, load_shard
from chessnet.encoding import ENCODING_VERSION
from chessnet.encoding.moves import index_to_move
from chessnet.encoding.packed import unpack_batch_torch, unpack_board

REPO = Path(__file__).parents[2]
FIXTURES = REPO / "tests" / "fixtures"
MINI = FIXTURES / "mini.pgn"
MINI_ZST = FIXTURES / "mini.pgn.zst"
CONFIG = DataConfig(shard_size=200, num_buckets=4, seed=7)
KEPT_GAMES = {f"white{number:02d}" for number in range(1, 12)}  # fixture games 01-11

EXPECTED_SPLITS = {Split.TRAIN: (9, 862), Split.VAL: (1, 95), Split.TEST: (1, 61)}
EXPECTED_REJECTIONS = {
    "elo": 1,
    "not_rated": 1,
    "result": 1,
    "termination": 2,
    "time_control": 2,
    "too_short": 1,
    "variant": 1,
}


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Manifest]:
    out = tmp_path_factory.mktemp("preprocess") / "shards"
    return out, preprocess(CONFIG, str(MINI_ZST), out, workers=2)


def stored_records(directory: Path, manifest: Manifest) -> list[np.void]:
    return [
        record
        for info in manifest.splits.values()
        for shard in info.shards
        for record in load_shard(directory / shard.file)
    ]


def test_counts_match_the_hand_checked_fixture(built: tuple[Path, Manifest]) -> None:
    _, manifest = built
    assert manifest.encoding_version == ENCODING_VERSION
    assert manifest.games.scanned == 20
    assert manifest.games.passed_header_filter == 12
    assert manifest.games.kept == 11
    assert manifest.games.rejected == EXPECTED_REJECTIONS
    assert manifest.skipped_positions == 0
    for split, (games, positions) in EXPECTED_SPLITS.items():
        assert (manifest.splits[split].games, manifest.splits[split].positions) == (
            games,
            positions,
        )


def test_manifest_describes_source_and_code(built: tuple[Path, Manifest]) -> None:
    _, manifest = built
    assert manifest.source.location == str(MINI_ZST)
    assert manifest.source.kind == "path"
    assert manifest.source.compressed
    assert manifest.source.bytes_read == MINI_ZST.stat().st_size
    assert manifest.source.bytes_decompressed == MINI.stat().st_size
    assert manifest.source.stopped_because == "end_of_source"
    assert manifest.config == CONFIG
    assert manifest.git_commit is not None
    assert re.fullmatch(r"[0-9a-f]{40}(-dirty)?", manifest.git_commit)


def test_shards_match_the_manifest(built: tuple[Path, Manifest]) -> None:
    directory, manifest = built
    assert not (directory / "buckets").exists()
    listed = {MANIFEST_NAME}
    for split, info in manifest.splits.items():
        assert sum(shard.positions for shard in info.shards) == info.positions
        for number, shard in enumerate(info.shards):
            assert shard.file == f"{split}-{number:05d}.npy"
            path = directory / shard.file
            assert len(load_shard(path)) == shard.positions
            assert hashlib.sha256(path.read_bytes()).hexdigest() == shard.sha256
            listed.add(shard.file)
    assert {path.name for path in directory.iterdir()} == listed


def expected_examples() -> Counter[tuple[str, str, int, int, int]]:
    """Replay the kept fixture games independently of the pipeline.

    One entry per position: (position before the move as the mover sees it, the move played
    from it in the mover's frame, WDL for the mover, ply, mover's rating).
    """
    wdl_by_result = {
        "1-0": {chess.WHITE: 0, chess.BLACK: 2},
        "0-1": {chess.WHITE: 2, chess.BLACK: 0},
        "1/2-1/2": {chess.WHITE: 1, chess.BLACK: 1},
    }
    examples: Counter[tuple[str, str, int, int, int]] = Counter()
    handle = io.StringIO(MINI.read_text(encoding="utf-8"))
    while (game := chess.pgn.read_game(handle)) is not None:
        if game.headers["White"] not in KEPT_GAMES:
            continue
        rating = {
            chess.WHITE: int(game.headers["WhiteElo"]),
            chess.BLACK: int(game.headers["BlackElo"]),
        }
        board = game.board()
        for move in game.mainline_moves():
            if board.turn == chess.WHITE:
                view, view_move = board, move
            else:
                view = board.mirror()
                view_move = chess.Move(
                    chess.square_mirror(move.from_square),
                    chess.square_mirror(move.to_square),
                    promotion=move.promotion,
                )
            wdl = wdl_by_result[game.headers["Result"]][board.turn]
            examples[(view.fen(), view_move.uci(), wdl, board.ply(), rating[board.turn])] += 1
            board.push(move)
    return examples


def test_every_stored_target_is_the_move_played_from_the_encoded_position(
    built: tuple[Path, Manifest],
) -> None:
    """Pipeline-level regression guard: the legacy dataset labelled each position with the
    move that led to it. Every stored record must hold the move played *from* the encoded
    position, legal in it, and the records must be exactly the positions of the kept games.
    """
    directory, manifest = built
    stored: Counter[tuple[str, str, int, int, int]] = Counter()
    for record in stored_records(directory, manifest):
        view = unpack_board(record)
        move = index_to_move(int(record["move"]), view)
        assert move in view.legal_moves, (view.fen(), move.uci())
        stored[
            (view.fen(), move.uci(), int(record["wdl"]), int(record["ply"]), int(record["elo"]))
        ] += 1
    assert stored == expected_examples()


def test_underpromotion_survives_the_pipeline(built: tuple[Path, Manifest]) -> None:
    directory, manifest = built
    promotions = [
        index_to_move(int(record["move"]), unpack_board(record)).promotion
        for record in stored_records(directory, manifest)
    ]
    assert chess.ROOK in promotions  # fixture game 08: ...h1=R


def test_runs_are_reproducible(built: tuple[Path, Manifest], tmp_path: Path) -> None:
    _, manifest = built
    again = preprocess(CONFIG, str(MINI_ZST), tmp_path / "again", workers=0)
    for split in Split:
        first = [shard.sha256 for shard in manifest.splits[split].shards]
        assert [shard.sha256 for shard in again.splits[split].shards] == first


def test_plain_text_source_gives_the_same_shards(
    built: tuple[Path, Manifest], tmp_path: Path
) -> None:
    _, manifest = built
    plain = preprocess(CONFIG, str(MINI), tmp_path / "plain", workers=0)
    assert not plain.source.compressed
    assert plain.splits == manifest.splits


@pytest.mark.parametrize("chunk_games", [64, 2])
def test_max_games_stops_the_run(tmp_path: Path, chunk_games: int) -> None:
    manifest = preprocess(
        CONFIG, str(MINI_ZST), tmp_path / "capped", max_games=3, workers=0, chunk_games=chunk_games
    )
    assert manifest.games.kept == 3
    assert manifest.source.stopped_because == "max_games"
    assert manifest.splits[Split.TRAIN].positions == 76 + 88 + 111  # games 01-03


def test_config_max_games_is_the_default(tmp_path: Path) -> None:
    config = CONFIG.model_copy(update={"max_games": 2})
    manifest = preprocess(config, str(MINI_ZST), tmp_path / "capped", workers=0)
    assert manifest.games.kept == 2


def test_output_directory_must_be_empty(tmp_path: Path) -> None:
    (tmp_path / "leftover.npy").write_bytes(b"")
    with pytest.raises(FileExistsError):
        preprocess(CONFIG, str(MINI_ZST), tmp_path, workers=0)


def test_unrepresentable_moves_fail_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = pipeline.encode_game

    def lossy(block: bytes, min_plies: int) -> pipeline.GameOutcome:
        outcome = original(block, min_plies)
        if outcome.rejection is None:
            return pipeline.GameOutcome(None, outcome.split, outcome.records, skipped_positions=1)
        return outcome

    monkeypatch.setattr(pipeline, "encode_game", lossy)
    with pytest.raises(PreprocessError, match="unrepresentable"):
        preprocess(CONFIG, str(MINI_ZST), tmp_path / "out", workers=0)
    assert not (tmp_path / "out" / MANIFEST_NAME).exists()


def test_progress_is_reported(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("INFO", logger="chessnet.data.pipeline"):
        preprocess(
            CONFIG,
            str(MINI_ZST),
            tmp_path / "out",
            max_games=100,
            workers=0,
            progress_every=0.0,
            chunk_games=4,
        )
    progress = [message for message in caplog.messages if message.startswith("read ")]
    assert len(progress) >= 3
    assert "scanned 20 games" in progress[-1]
    assert any("ETA" in message for message in progress[:-1])
    assert "ETA" not in progress[-1]  # the final summary has nothing left to estimate


def test_loader_yields_a_batch_with_the_right_shapes(built: tuple[Path, Manifest]) -> None:
    directory, _ = built
    batch = next(iter(make_loader(ShardDataset(directory, batch_size=32), num_workers=0)))
    assert batch.bitboards.shape == (32, 12)
    assert batch.bitboards.dtype == torch.int64
    assert batch.move.shape == batch.wdl.shape == (32,)
    planes = unpack_batch_torch(batch)
    assert planes.shape == (32, 19, 8, 8)
    assert planes.dtype == torch.float32


def test_cli_preprocess(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "preprocess",
            "--config",
            str(REPO / "configs" / "data" / "lichess_2000plus.yaml"),
            "--source",
            str(MINI_ZST),
            "--out",
            str(tmp_path / "out"),
            "--workers",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "train: 9 games, 862 positions, 1 shards" in result.output
    assert (tmp_path / "out" / MANIFEST_NAME).exists()


def test_default_worker_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline.os, "cpu_count", lambda: None)  # unknown: one worker
    manifest = preprocess(CONFIG, str(MINI_ZST), tmp_path / "out")
    assert manifest.games.kept == 11
