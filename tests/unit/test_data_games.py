import io
from pathlib import Path

import chess
import pytest

from chessnet.data import pipeline
from chessnet.data.filters import Rejection
from chessnet.data.pgn_stream import iter_game_blocks
from chessnet.data.shards import Split
from chessnet.encoding.moves import move_to_index
from chessnet.encoding.value import DRAW, LOSS, WIN

FIXTURES = Path(__file__).parents[1] / "fixtures"
BLOCKS = list(iter_game_blocks(io.BytesIO((FIXTURES / "mini.pgn").read_bytes()).read))


def game(number: int) -> bytes:
    return BLOCKS[number - 1]


def test_every_position_is_packed_with_the_move_played_from_it() -> None:
    outcome = pipeline.encode_game(game(1), min_plies=20)  # 1-0, 76 plies, 2103 vs 2125
    assert outcome.rejection is None
    assert outcome.split == Split.TRAIN
    assert outcome.skipped_positions == 0
    records = outcome.records
    assert records is not None
    assert len(records) == 76
    first, second = records[0], records[1]
    assert first["move"] == move_to_index(chess.Move.from_uci("d2d4"), chess.Board())
    assert (first["wdl"], first["elo"], first["ply"]) == (WIN, 2103, 0)
    after_d4 = chess.Board()
    after_d4.push_uci("d2d4")
    assert second["move"] == move_to_index(chess.Move.from_uci("g8f6"), after_d4)
    assert (second["wdl"], second["elo"], second["ply"]) == (LOSS, 2125, 1)


def test_draws_and_black_wins() -> None:
    draw = pipeline.encode_game(game(3), min_plies=20).records
    black_win = pipeline.encode_game(game(2), min_plies=20).records
    assert draw is not None
    assert black_win is not None
    assert set(draw["wdl"].tolist()) == {DRAW}
    assert black_win["wdl"][0] == LOSS  # White to move in a game Black won
    assert black_win["wdl"][1] == WIN


def test_split_comes_from_the_game_url() -> None:
    assert pipeline.encode_game(game(9), min_plies=20).split == Split.VAL
    assert pipeline.encode_game(game(10), min_plies=20).split == Split.TEST


def test_short_games_are_rejected() -> None:
    assert pipeline.encode_game(game(15), min_plies=20).rejection == Rejection.TOO_SHORT
    assert pipeline.encode_game(game(15), min_plies=13).rejection is None


def test_illegal_moves_are_parse_errors() -> None:
    broken = game(1).replace(b"1. d4 ", b"1. d5 ", 1)
    assert pipeline.encode_game(broken, min_plies=20).rejection == Rejection.PARSE_ERROR


def test_text_without_a_game_is_malformed() -> None:
    assert pipeline.encode_game(b"", min_plies=20).rejection == Rejection.MALFORMED


def test_encode_games_keeps_order() -> None:
    outcomes = pipeline.encode_games([game(1), game(15), game(9)], min_plies=20)
    assert [outcome.rejection for outcome in outcomes] == [None, Rejection.TOO_SHORT, None]
    assert [outcome.split for outcome in outcomes] == [Split.TRAIN, None, Split.VAL]


def test_unrepresentable_moves_are_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    real_pack_into = pipeline.pack_into
    calls = 0

    def flaky_pack_into(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 5:
            raise ValueError("pretend this move is unrepresentable")
        real_pack_into(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pipeline, "pack_into", flaky_pack_into)
    outcome = pipeline.encode_game(game(1), min_plies=20)
    assert outcome.skipped_positions == 1
    assert outcome.records is not None
    assert len(outcome.records) == 75


def test_variations_are_ignored() -> None:
    block = (
        b'[Event "Rated Blitz game"]\n[Site "https://lichess.org/game/variation"]\n'
        b'[Result "1-0"]\n[WhiteElo "2100"]\n[BlackElo "2000"]\n\n'
        b"1. e4 (1. d4 d5 2. c4) 1... e5 2. Nf3 (2. Bc4 Nf6) 2... Nc6 1-0\n"
    )
    outcome = pipeline.encode_game(block, min_plies=1)
    assert outcome.records is not None
    board = chess.Board()
    expected = []
    for san in ("e4", "e5", "Nf3", "Nc6"):
        move = board.parse_san(san)
        expected.append(move_to_index(move, board))
        board.push(move)
    assert outcome.records["move"].tolist() == expected
