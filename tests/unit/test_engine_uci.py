"""The UCI engine in-process: protocol parsing, options, time management, scores, searches."""

import itertools
import threading
import time
from pathlib import Path

import chess
import pytest

from chessnet.engine.timeman import move_time
from chessnet.engine.uci import (
    NAME,
    GoLimits,
    Options,
    UCIEngine,
    depth_to_nodes,
    info_line,
    parse_go,
    parse_position,
    parse_setoption,
    value_to_cp,
)

from helpers.search import tiny_checkpoint

BLACK_TO_MOVE = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"


def tokens(text: str) -> list[str]:
    return text.split()


# --- protocol parsing ------------------------------------------------------------------


def test_parse_position() -> None:
    assert parse_position(["startpos"]) == chess.Board()
    board = parse_position(["startpos", "moves", "e2e4", "e7e5", "g1f3"])
    assert [move.uci() for move in board.move_stack] == ["e2e4", "e7e5", "g1f3"]
    assert board.turn == chess.BLACK
    fen = parse_position(["fen", *BLACK_TO_MOVE.split()])
    assert fen.fen() == BLACK_TO_MOVE
    played = parse_position(["fen", *BLACK_TO_MOVE.split(), "moves", "b8c6", "f1b5"])
    assert played.move_stack[0] == chess.Move.from_uci("b8c6")  # the history is kept
    assert played.root().fen() == BLACK_TO_MOVE
    castled = parse_position(["fen", "r3k2r/8/8/8/8/8/8/R3K2R", "w", "KQkq", "-", "moves", "e1g1"])
    assert castled.piece_at(chess.G1) == chess.Piece(chess.KING, chess.WHITE)


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ([], "cannot read the position"),
        (["startpos", "extra"], "cannot read the position"),
        (["fen"], "cannot read the position"),
        (["fen", "not", "a", "fen"], "fen"),
        (["startpos", "moves", "e2e5"], "illegal move e2e5"),
        (["startpos", "moves", "zz"], "uci"),
    ],
)
def test_parse_position_refuses_nonsense(args: list[str], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        parse_position(args)


def test_parse_go() -> None:
    limits = parse_go(tokens("wtime 61000 btime 59000 winc 1000 binc 1000 movestogo 40"))
    assert limits == GoLimits(wtime=61.0, btime=59.0, winc=1.0, binc=1.0, movestogo=40)
    assert parse_go(["infinite"]).infinite
    assert parse_go(tokens("nodes 800 movetime 2500")) == GoLimits(nodes=800, movetime=2.5)
    assert parse_go(tokens("searchmoves e2e4 d2d4 nodes 5")) == GoLimits(nodes=5)
    assert parse_go(tokens("ponder depth 7 mate 3 whatever")) == GoLimits(depth=7)
    assert parse_go(tokens("wtime -150")).wtime == 0.0  # some GUIs send negative clocks
    assert parse_go([]) == GoLimits()


def test_parse_setoption() -> None:
    assert parse_setoption(tokens("name Weights value /a b/c.pt")) == ("Weights", "/a b/c.pt")
    assert parse_setoption(tokens("name Move Overhead value 100")) == ("Move Overhead", "100")
    assert parse_setoption(tokens("name Clear Hash")) == ("Clear Hash", "")
    with pytest.raises(ValueError, match="needs a name"):
        parse_setoption(["value", "3"])


def test_options_are_validated() -> None:
    options = Options(weights="w.pt")
    options.set("simulations", "1600")
    options.set("CPUCT", "1.5")
    options.set("Mode", "POLICY")
    options.set("LeafBatch", "16")
    options.set("Device", "cpu")
    options.set("MoveOverhead", "250")
    options.set("Weights", "<empty>")
    assert (options.simulations, options.c_puct, options.mode) == (1600, 1.5, "policy")
    assert (options.leaf_batch, options.device, options.move_overhead) == (16, "cpu", 250)
    assert options.weights == ""
    for name, value, error in [
        ("Simulations", "0", "outside"),
        ("CPuct", "-1", "positive"),
        ("Mode", "alphabeta", "not one of"),
        ("LeafBatch", "1000", "outside"),
        ("Device", "tpu", "not one of"),
        ("Hash", "128", "unknown option"),
        ("Simulations", "many", "invalid literal"),
    ]:
        with pytest.raises(ValueError, match=error):
            options.set(name, value)


# --- scores and time -------------------------------------------------------------------


def test_value_to_cp_follows_the_lichess_model() -> None:
    assert value_to_cp(0.0) == 0
    assert value_to_cp(0.5) == -value_to_cp(-0.5) == 298  # ln(3) / 0.00368208
    assert 95 <= value_to_cp(0.18) <= 105  # about a pawn
    values = [value_to_cp(v / 10) for v in range(-10, 11)]
    assert values == sorted(values)
    assert value_to_cp(1.0) == value_to_cp(0.999) < 2100  # clipped


def test_move_time_formula_and_caps() -> None:
    assert move_time(60.0, 0.0, overhead=0.0) == pytest.approx(2.0)  # 60 / 30
    assert move_time(60.0, 1.0, overhead=0.0) == pytest.approx(2.8)  # + 0.8 * increment
    assert move_time(60.0, 1.0) == pytest.approx(2.7)  # less the 0.1 s overhead
    assert move_time(10.0, 0.0, moves_to_go=5, overhead=0.0) == pytest.approx(2.0)
    assert move_time(1.0, 5.0) == pytest.approx(0.45)  # capped at half of what is left
    assert move_time(10.0, 0.0, moves_to_go=1) == pytest.approx(4.95)
    assert move_time(0.05, 0.0) == 0.0
    assert move_time(0.0, 0.0) == 0.0


@pytest.mark.parametrize(
    ("base", "increment"),
    [(10, 0.1), (30, 0), (60, 0), (60, 1), (120, 0), (180, 0), (180, 2), (300, 0)],
)
def test_short_time_controls_never_flag(base: float, increment: float) -> None:
    """A pessimistic model of each move's cost: the budget, plus one CPU leaf batch past the
    deadline (60 ms) and 20 ms of latency; with no budget, one evaluation and the latency."""
    remaining = float(base)
    for _ in range(100):  # a long game
        budget = move_time(remaining, increment)
        spent = budget + 0.08 if budget > 0 else 0.05
        remaining -= spent
        assert remaining > 0
        remaining += increment


def test_info_line() -> None:
    from chessnet.search.mcts import SearchResult

    e4, e5 = chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5")
    result = SearchResult(
        move=e4,
        visits={e4: 90},
        value=0.1,
        move_value=0.18,
        mate_in_one=False,
        pv=[e4, e5],
        playouts=500,
        nodes=600,
        seconds=0.5,
    )
    line = info_line(result)
    assert line.startswith("info depth 2 score cp ")
    assert " nodes 600 nps 1000 time 500 pv e2e4 e7e5" in line
    mate = SearchResult(e4, {e4: 1}, 1.0, 1.0, True, [], 1, 1, 0.0)
    assert info_line(mate).startswith("info depth 1 score mate 1 ")
    assert info_line(mate).endswith("pv e2e4")


# --- the engine ------------------------------------------------------------------------


class Session:
    """An engine fed command lines, collecting its output."""

    def __init__(self, weights: Path | str) -> None:
        self.lines: list[str] = []
        self.bestmove = threading.Event()
        self.engine = UCIEngine(self._receive, weights=str(weights), device="cpu")

    def _receive(self, line: str) -> None:
        self.lines.append(line)
        if line.startswith("bestmove"):
            self.bestmove.set()

    def send(self, *commands: str) -> None:
        for command in commands:
            assert self.engine.handle(command)

    def best_move(self, timeout: float = 60.0) -> str:
        assert self.bestmove.wait(timeout)
        self.bestmove.clear()
        return next(line for line in reversed(self.lines) if line.startswith("bestmove")).split()[1]


@pytest.fixture(scope="module")
def weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny_checkpoint(tmp_path_factory.mktemp("weights") / "tiny.pt")


def test_handshake(weights: Path) -> None:
    session = Session(weights)
    session.send("uci")
    assert session.lines[0] == f"id name {NAME}"
    assert session.lines[-1] == "uciok"
    options = [line.split()[2] for line in session.lines if line.startswith("option name")]
    assert options == [
        "Weights",
        "Simulations",
        "CPuct",
        "Mode",
        "LeafBatch",
        "Device",
        "MoveOverhead",
    ]
    assert f"option name Weights type string default {weights}" in session.lines
    session.send("isready")
    assert session.lines[-1] == "readyok"
    assert any(line.startswith("info string loaded") for line in session.lines)


@pytest.mark.parametrize("position", ["startpos moves e2e4", f"fen {BLACK_TO_MOVE}", "startpos"])
def test_go_nodes_plays_a_legal_move(weights: Path, position: str) -> None:
    session = Session(weights)
    session.send(f"position {position}", "go nodes 40")
    board = parse_position(position.split())
    assert chess.Move.from_uci(session.best_move()) in board.legal_moves
    info = [line for line in session.lines if line.startswith("info depth")]
    assert info
    assert " score cp " in info[-1] or " score mate " in info[-1]
    assert " nodes " in info[-1]
    assert " pv " in info[-1]


def test_stop_ends_an_infinite_search(weights: Path) -> None:
    session = Session(weights)
    session.send("position startpos", "go infinite")
    time.sleep(0.5)
    assert not session.bestmove.is_set()  # no bestmove before stop in infinite mode
    started = time.monotonic()
    session.send("stop")
    move = session.best_move(timeout=5)
    assert time.monotonic() - started < 2.0
    assert chess.Move.from_uci(move) in chess.Board().legal_moves


def test_isready_answers_during_a_search(weights: Path) -> None:
    session = Session(weights)
    session.send("isready", "position startpos", "go infinite")
    time.sleep(0.2)
    session.send("isready")
    assert session.lines[-1] == "readyok"
    session.send("stop")
    session.best_move(timeout=5)


def test_clock_limits_bound_the_search(weights: Path) -> None:
    session = Session(weights)
    session.send("position startpos")
    started = time.monotonic()
    session.send("go wtime 3000 btime 3000")  # 3 s / 30 - 0.1 s overhead: 0 s to spare
    session.best_move(timeout=10)
    assert time.monotonic() - started < 1.0
    session.send("go movetime 400")
    started = time.monotonic()
    session.best_move(timeout=10)
    assert 0.2 < time.monotonic() - started < 2.0


def test_policy_mode_plays_the_networks_move(weights: Path) -> None:
    session = Session(weights)
    session.send("setoption name Mode value policy", "position startpos", "go")
    move = session.best_move()
    assert chess.Move.from_uci(move) in chess.Board().legal_moves
    assert any(line.startswith("info depth 1 score cp") for line in session.lines)


def test_bad_input_is_reported_and_ignored(weights: Path) -> None:
    session = Session(weights)
    session.send("position startpos moves e2e4")
    session.send("position startpos moves e2e5")  # illegal: the position is kept
    assert session.lines[-1].startswith("info string illegal move e2e5")
    assert session.engine.board.move_stack == [chess.Move.from_uci("e2e4")]
    session.send("setoption name LeafBatch value 0", "frobnicate", "debug on", "")
    assert session.lines[-1].startswith("info string 0 is outside")
    session.send("go nodes x")
    assert chess.Move.from_uci(session.best_move()) in session.engine.board.legal_moves
    assert not session.engine.handle("quit")


def test_missing_weights_fail_clearly(tmp_path: Path) -> None:
    session = Session(tmp_path / "missing.pt")
    session.send("isready")
    assert session.lines[-2].startswith("info string no weights at")
    assert session.lines[-1] == "readyok"
    session.send("position startpos", "go nodes 10")
    assert session.best_move() == "0000"


def test_a_finished_game_gets_a_null_move(weights: Path) -> None:
    session = Session(weights)
    session.send("position fen R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1", "go nodes 10")
    assert session.best_move() == "0000"
    assert any("no legal moves" in line for line in session.lines)


def test_the_tree_is_reused_and_ucinewgame_drops_it(weights: Path) -> None:
    session = Session(weights)
    session.send("position startpos", "go nodes 200")
    first = session.best_move()
    session.send(f"position startpos moves {first}", "go nodes 50")
    session.best_move()
    reused = int(
        next(line for line in reversed(session.lines) if " nodes " in line)
        .split(" nodes ")[1]
        .split()[0]
    )
    assert reused > 50  # the subtree of the move played was kept
    session.send("ucinewgame", f"position startpos moves {first}", "go nodes 50")
    session.best_move()
    fresh = int(
        next(line for line in reversed(session.lines) if " nodes " in line)
        .split(" nodes ")[1]
        .split()[0]
    )
    assert fresh == 50


def last_nodes(session: Session) -> int:
    line = next(line for line in reversed(session.lines) if " nodes " in line)
    return int(line.split(" nodes ")[1].split()[0])


@pytest.mark.parametrize("depth", [1, 3, 9])
def test_go_depth_limits_the_search(weights: Path, depth: int) -> None:
    """A GUI's depth setting must change the search, not be ignored (it was)."""
    session = Session(weights)
    session.send("position startpos", f"go depth {depth}")
    session.best_move()
    assert last_nodes(session) == depth_to_nodes(depth)


def test_depth_to_nodes_grows_with_depth() -> None:
    assert [depth_to_nodes(depth) for depth in (1, 2, 5, 9, 12, 20)] == [
        10,
        40,
        250,
        810,
        1440,
        4000,
    ]
    budgets = [depth_to_nodes(depth) for depth in range(1, 400)]
    assert all(a < b for a, b in itertools.pairwise(budgets) if b < 1_000_000)
    assert budgets == sorted(budgets)
    assert depth_to_nodes(10_000) == 1_000_000  # capped


@pytest.mark.parametrize(
    ("fen", "mate"),
    [
        ("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", "d1d8"),
        ("3r2k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1", "d8d1"),
    ],
)
def test_policy_mode_plays_policy_safe(weights: Path, fen: str, mate: str) -> None:
    """Mode policy takes a mate in one, whatever the (random, tiny) network prefers."""
    session = Session(weights)
    session.send("setoption name Mode value policy", f"position fen {fen}", "go")
    assert session.best_move() == mate
