"""The engine as a GUI runs it: a subprocess speaking UCI, driven by python-chess."""

import os
import shutil
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import chess
import chess.engine
import pytest

from helpers.search import tiny_checkpoint

BLACK_TO_MOVE = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
OPTIONS = {"Weights", "Simulations", "CPuct", "Mode", "LeafBatch", "Device", "MoveOverhead"}


@pytest.fixture(scope="module")
def weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny_checkpoint(tmp_path_factory.mktemp("weights") / "tiny.pt")


def start(weights: Path) -> chess.engine.SimpleEngine:
    command = [
        sys.executable,
        "-m",
        "chessnet",
        "uci",
        "--weights",
        str(weights),
        "--device",
        "cpu",
    ]
    return chess.engine.SimpleEngine.popen_uci(command, timeout=60)


@pytest.fixture
def engine(weights: Path) -> Iterator[chess.engine.SimpleEngine]:
    engine = start(weights)
    try:
        yield engine
    finally:
        engine.quit()


def test_it_identifies_itself_and_its_options(engine: chess.engine.SimpleEngine) -> None:
    assert engine.id["name"].startswith("chessnet ")
    assert set(engine.options) >= OPTIONS
    assert engine.options["Mode"].var == ["mcts", "policy"]


def test_a_legal_move_from_the_start_position(engine: chess.engine.SimpleEngine) -> None:
    board = chess.Board()
    result = engine.play(board, chess.engine.Limit(nodes=50), info=chess.engine.INFO_ALL)
    assert result.move in board.legal_moves
    assert {"score", "nodes", "nps", "pv", "depth"} <= set(result.info)


def test_a_legal_move_with_black_to_move(engine: chess.engine.SimpleEngine) -> None:
    board = chess.Board(BLACK_TO_MOVE)
    result = engine.play(board, chess.engine.Limit(nodes=50))
    assert result.move in board.legal_moves
    after_moves = chess.Board()
    for san in ("e4", "e5", "Nf3"):
        after_moves.push_san(san)
    result = engine.play(after_moves, chess.engine.Limit(nodes=50))
    assert result.move in after_moves.legal_moves


def test_stop_ends_infinite_analysis(engine: chess.engine.SimpleEngine) -> None:
    board = chess.Board()
    with engine.analysis(board) as analysis:  # go infinite
        time.sleep(1.5)
        assert analysis.info.get("pv")  # progress reports arrive while it searches
        started = time.monotonic()
        analysis.stop()
        best = analysis.wait()
    assert time.monotonic() - started < 2.0
    assert best.move in board.legal_moves


def test_options_change_the_search(engine: chess.engine.SimpleEngine) -> None:
    engine.configure({"Mode": "policy"})
    board = chess.Board()
    result = engine.play(board, chess.engine.Limit(nodes=500), info=chess.engine.INFO_ALL)
    assert result.move in board.legal_moves
    assert result.info["nodes"] == 1  # no search
    engine.configure({"Mode": "mcts", "Simulations": 30, "LeafBatch": 8, "CPuct": "1.5"})
    result = engine.play(board, chess.engine.Limit(), info=chess.engine.INFO_ALL)
    assert result.info["nodes"] >= 30


def play_on_the_clock(
    engine: chess.engine.SimpleEngine, base: float, increment: float, plies: int
) -> None:
    """Both sides on the clock; each move's real time, measured from the GUI's side, comes
    off the mover's clock, which must never run out."""
    clocks = {chess.WHITE: base, chess.BLACK: base}
    board = chess.Board()
    for _ in range(plies):
        if board.outcome(claim_draw=True) is not None:
            break
        limit = chess.engine.Limit(
            white_clock=clocks[chess.WHITE],
            black_clock=clocks[chess.BLACK],
            white_inc=increment,
            black_inc=increment,
        )
        started = time.monotonic()
        result = engine.play(board, limit)
        clocks[board.turn] -= time.monotonic() - started
        assert clocks[board.turn] > 0, f"flagged at ply {board.ply()}"
        clocks[board.turn] += increment
        assert result.move in board.legal_moves
        board.push(result.move)


def test_a_short_time_control_never_flags(engine: chess.engine.SimpleEngine) -> None:
    play_on_the_clock(engine, base=5.0, increment=0.05, plies=40)


@pytest.mark.slow
def test_a_longer_blitz_game_never_flags(engine: chess.engine.SimpleEngine) -> None:
    play_on_the_clock(engine, base=20.0, increment=0.0, plies=120)


def test_chessnet_uci_runs_without_arguments(weights: Path, tmp_path: Path) -> None:
    script = shutil.which("chessnet-uci")
    if script is None:
        pytest.skip("the chessnet-uci script is not installed")
    environment = {**os.environ, "CHESSNET_WEIGHTS": str(weights)}
    engine = chess.engine.SimpleEngine.popen_uci([script], env=environment, cwd=tmp_path)
    try:
        assert engine.options["Weights"].default == str(weights)
        board = chess.Board()
        assert engine.play(board, chess.engine.Limit(nodes=20)).move in board.legal_moves
    finally:
        engine.quit()
