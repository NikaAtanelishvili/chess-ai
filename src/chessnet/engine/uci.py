"""UCI engine: the protocol, the options, and the search running in a worker thread.

Chess GUIs start ``chessnet-uci`` (or ``chessnet uci``) and speak UCI over stdin and stdout.
Heavy imports and the network wait until they are needed: ``uci`` is answered at once,
and ``isready`` loads the weights. Searches run in a worker thread, so ``stop``,
``isready`` and ``quit`` are answered while the engine thinks.
"""

from __future__ import annotations

import math
import os
import sys
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import chess

from chessnet import __version__
from chessnet.engine.timeman import MOVE_OVERHEAD, move_time

if TYPE_CHECKING:
    from chessnet.search.evaluator import NetworkEvaluator
    from chessnet.search.mcts import MCTS, SearchResult

NAME = f"chessnet {__version__}"
AUTHOR = "Nikoloz Atanelishvili"
#: The chosen network (docs/TRAINING.md) where it lives in a checkout of the repository.
REPOSITORY_WEIGHTS = (
    Path(__file__).resolve().parents[3] / "checkpoints" / "session2" / "base-best.pt"
)
#: Lichess's model of winning chances, ``v = 2 / (1 + exp(-k * cp)) - 1``, gives the slope.
CP_SLOPE = 0.00368208
#: ``|v|`` is clipped to this before conversion, so scores stay within about 2,000 cp.
MAX_ABS_VALUE = 0.999
#: Measured memory of the search tree per playout is about 9 KB.
BYTES_PER_NODE = 10_000
#: Positions the evaluator caches (a few kilobytes each).
CACHE_SIZE = 20_000
MODES = ("mcts", "policy")
DEVICES = ("auto", "cpu", "cuda")


def default_weights() -> str:
    """``CHESSNET_WEIGHTS`` if set, else the chosen network in this repository."""
    return os.environ.get("CHESSNET_WEIGHTS") or str(REPOSITORY_WEIGHTS)


def value_to_cp(value: float) -> int:
    """Centipawns for an expected outcome ``value`` in [-1, 1] (Lichess's logistic model).

    ``cp = ln((1 + v) / (1 - v)) / 0.00368208``: v = 0.18 is about one pawn, 0.5 about three.
    """
    clipped = max(-MAX_ABS_VALUE, min(MAX_ABS_VALUE, value))
    return round(math.log((1 + clipped) / (1 - clipped)) / CP_SLOPE)


#: ``go depth N`` searches ``DEPTH_NODES * N**2`` playouts, capped here.
DEPTH_NODES = 10
MAX_DEPTH_NODES = 1_000_000


def depth_to_nodes(depth: int) -> int:
    """The playout budget for ``go depth N``: ``10 * N**2``, capped at a million.

    MCTS has no fixed depth, so a depth setting becomes a node budget that grows with it:
    depth 1 is 10 playouts, depth 9 is 810 (about the default), depth 20 is 4,000.
    """
    return min(DEPTH_NODES * depth * depth, MAX_DEPTH_NODES)


def node_budget() -> int:
    """Root visits a tree may hold: half the memory available at start-up, at ~10 KB each."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                return max(20_000, min(available // 2 // BYTES_PER_NODE, 2_000_000))
    except (OSError, ValueError):
        pass
    return 200_000


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Options:
    """The engine's UCI options."""

    weights: str
    simulations: int = 800  # playouts per move when "go" gives no limit
    c_puct: float = 2.0
    mode: str = "mcts"  # "mcts" or "policy" (the network's top move, no search)
    leaf_batch: int = 32  # the fastest on the CPUs measured (docs/ARCHITECTURE.md)
    device: str = "auto"
    move_overhead: int = round(MOVE_OVERHEAD * 1000)  # milliseconds

    def uci_lines(self) -> list[str]:
        return [
            f"option name Weights type string default {self.weights}",
            f"option name Simulations type spin default {self.simulations} min 1 max 10000000",
            f"option name CPuct type string default {self.c_puct}",
            f"option name Mode type combo default {self.mode} var mcts var policy",
            f"option name LeafBatch type spin default {self.leaf_batch} min 1 max 256",
            f"option name Device type combo default {self.device} var auto var cpu var cuda",
            f"option name MoveOverhead type spin default {self.move_overhead} min 0 max 5000",
        ]

    def set(self, name: str, value: str) -> None:
        """Set option ``name`` (any case) from its UCI text; ValueError if invalid."""
        key = name.lower()
        if key == "weights":
            self.weights = "" if value == "<empty>" else value
        elif key == "simulations":
            self.simulations = _bounded_int(value, 1, 10_000_000)
        elif key == "cpuct":
            number = float(value)
            if not number > 0:
                raise ValueError(f"CPuct must be positive, not {value}")
            self.c_puct = number
        elif key == "mode":
            self.mode = _choice(value, MODES)
        elif key == "leafbatch":
            self.leaf_batch = _bounded_int(value, 1, 256)
        elif key == "device":
            self.device = _choice(value, DEVICES)
        elif key == "moveoverhead":
            self.move_overhead = _bounded_int(value, 0, 5000)
        else:
            raise ValueError(f"unknown option {name}")


def _bounded_int(value: str, low: int, high: int) -> int:
    number = int(value)
    if not low <= number <= high:
        raise ValueError(f"{number} is outside {low}..{high}")
    return number


def _choice(value: str, choices: tuple[str, ...]) -> str:
    if value.lower() not in choices:
        raise ValueError(f"{value} is not one of {', '.join(choices)}")
    return value.lower()


@dataclass(frozen=True)
class GoLimits:
    """The limits of a ``go`` command; times in seconds."""

    nodes: int | None = None
    movetime: float | None = None
    wtime: float | None = None
    btime: float | None = None
    winc: float = 0.0
    binc: float = 0.0
    movestogo: int | None = None
    infinite: bool = False
    depth: int | None = None  # becomes a playout budget: depth_to_nodes


_MILLISECONDS = ("wtime", "btime", "winc", "binc", "movetime")
_COUNTS = ("nodes", "movestogo", "depth")
GO_KEYWORDS = frozenset((*_MILLISECONDS, *_COUNTS, "mate", "searchmoves", "ponder", "infinite"))


def parse_go(args: list[str]) -> GoLimits:
    """Parse the tokens after ``go``. Unknown tokens are ignored, as UCI asks."""
    values: dict[str, float | int | bool] = {}
    index = 0
    while index < len(args):
        token = args[index]
        if token == "infinite":
            values["infinite"] = True
        elif token == "searchmoves":  # not supported: skip its moves
            while index + 1 < len(args) and args[index + 1] not in GO_KEYWORDS:
                index += 1
        elif token in _MILLISECONDS + _COUNTS + ("mate",) and index + 1 < len(args):
            index += 1
            number = int(args[index])
            if token in _MILLISECONDS:
                values[token] = max(number, 0) / 1000  # some GUIs send negative clocks
            elif token in _COUNTS:
                values[token] = max(number, 1)
        index += 1
    return GoLimits(**values)  # type: ignore[arg-type]


def parse_position(args: list[str]) -> chess.Board:
    """``startpos|fen <fen> [moves <uci> ...]``: the board, with the moves on its stack."""
    if "moves" in args:
        split = args.index("moves")
        setup, moves = args[:split], args[split + 1 :]
    else:
        setup, moves = args, []
    if setup == ["startpos"]:
        board = chess.Board()
    elif len(setup) > 1 and setup[0] == "fen":
        board = chess.Board(" ".join(setup[1:]))
    else:
        raise ValueError(f"cannot read the position {' '.join(args)!r}")
    for text in moves:
        move = chess.Move.from_uci(text)
        if move not in board.legal_moves:
            raise ValueError(f"illegal move {text} in {board.fen()}")
        board.push(move)
    return board


def parse_setoption(args: list[str]) -> tuple[str, str]:
    """``name <name> [value <value>]`` (both may contain spaces)."""
    if not args or args[0] != "name":
        raise ValueError("setoption needs a name")
    rest = args[1:]
    if "value" in rest:
        split = rest.index("value")
        return " ".join(rest[:split]), " ".join(rest[split + 1 :])
    return " ".join(rest), ""


def info_line(result: SearchResult) -> str:
    """``info depth .. score .. nodes .. nps .. time .. pv ..`` for a search result.

    Depth is the length of the principal variation. A move that checkmates is reported
    as ``score mate 1``; otherwise the chosen move's value in centipawns.
    """
    pv = result.pv or [result.move]
    score = "mate 1" if result.mate_in_one else f"cp {value_to_cp(result.move_value)}"
    return (
        f"info depth {len(pv)} score {score} nodes {result.nodes} "
        f"nps {round(result.nodes_per_second)} time {round(result.seconds * 1000)} "
        f"pv {' '.join(move.uci() for move in pv)}"
    )


class UCIEngine:
    """Executes UCI commands, one line at a time; ``send`` writes one line of output."""

    def __init__(
        self,
        send: Callable[[str], None],
        *,
        weights: str | None = None,
        device: str | None = None,
    ) -> None:
        self._write = send
        self._output_lock = threading.Lock()
        self.options = Options(weights=weights or default_weights())
        if device is not None:
            self.options.device = _choice(device, DEVICES)
        self.board = chess.Board()
        self.max_nodes = node_budget()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._evaluator: NetworkEvaluator | None = None
        self._loaded: tuple[str, str] | None = None
        self._mcts: MCTS | None = None
        self._mcts_settings: tuple[float, int] | None = None

    def send(self, line: str) -> None:
        with self._output_lock:
            self._write(line)

    def handle(self, line: str) -> bool:
        """Execute one command line; returns False after ``quit``."""
        tokens = line.split()
        if not tokens:
            return True
        command, args = tokens[0], tokens[1:]
        if command == "uci":
            self.send(f"id name {NAME}")
            self.send(f"id author {AUTHOR}")
            for option in self.options.uci_lines():
                self.send(option)
            self.send("uciok")
        elif command == "isready":
            if not self._searching():
                self._load()
            self.send("readyok")
        elif command == "setoption":
            self.stop_search()
            try:
                self.options.set(*parse_setoption(args))
            except ValueError as error:
                self.send(f"info string {error}")
        elif command == "ucinewgame":
            self.stop_search()
            self._mcts = None
            if self._evaluator is not None:
                self._evaluator.clear_cache()
        elif command == "position":
            self.stop_search()
            try:
                self.board = parse_position(args)
            except ValueError as error:
                self.send(f"info string {error}")
        elif command == "go":
            self.stop_search()
            try:
                limits = parse_go(args)
            except ValueError as error:
                self.send(f"info string {error}: searching with the default limits")
                limits = GoLimits()
            self._start_search(limits)
        elif command == "stop":
            self.stop_search()
        elif command == "quit":
            self.stop_search()
            return False
        return True  # anything else (debug, register, ponderhit, ...) is ignored

    def stop_search(self) -> None:
        """Stop a running search and wait for its ``bestmove``."""
        worker = self._worker
        if worker is not None:
            self._stop.set()
            worker.join()
            self._worker = None

    def _searching(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def _start_search(self, limits: GoLimits) -> None:
        self._stop.clear()
        self._worker = threading.Thread(
            target=self._search, args=(self.board.copy(), limits), name="search", daemon=True
        )
        self._worker.start()

    def _search(self, board: chess.Board, limits: GoLimits) -> None:
        move: chess.Move | None = None
        try:
            move = self._think(board, limits)
        except Exception as error:  # report it; the engine must keep running
            self.send(f"info string search failed: {type(error).__name__}: {error}")
        if limits.infinite:
            self._stop.wait()  # in infinite mode, bestmove only after stop
        self.send(f"bestmove {move.uci() if move else '0000'}")

    def _think(self, board: chess.Board, limits: GoLimits) -> chess.Move | None:
        if not any(board.generate_legal_moves()):
            self.send("info string no legal moves: the game is over")
            return None
        evaluator = self._load()
        if evaluator is None:
            return None
        if self.options.mode == "policy":
            [evaluation] = evaluator.evaluate([board])
            move = max(evaluation.priors, key=evaluation.priors.__getitem__)
            cp = value_to_cp(evaluation.value)
            self.send(f"info depth 1 score cp {cp} nodes 1 pv {move.uci()}")
            return move
        simulations, movetime = self._budget(board, limits)
        result = self._tree(evaluator).search(
            board,
            simulations=simulations,
            movetime=movetime,
            stop=self._stop,
            max_nodes=self.max_nodes,
            on_progress=lambda interim: self.send(info_line(interim)),
        )
        self.send(info_line(result))
        return result.move

    def _budget(self, board: chess.Board, limits: GoLimits) -> tuple[int | None, float | None]:
        """Playouts and seconds for this move; (None, None) searches until stop."""
        if limits.infinite:
            return None, None
        overhead = self.options.move_overhead / 1000
        movetime = None if limits.movetime is None else max(limits.movetime - overhead, 0.0)
        white = board.turn == chess.WHITE
        clock = limits.wtime if white else limits.btime
        if clock is not None:
            increment = limits.winc if white else limits.binc
            managed = move_time(clock, increment, moves_to_go=limits.movestogo, overhead=overhead)
            movetime = managed if movetime is None else min(movetime, managed)
        simulations = limits.nodes
        if limits.depth is not None:
            by_depth = depth_to_nodes(limits.depth)
            simulations = by_depth if simulations is None else min(simulations, by_depth)
        if simulations is None and movetime is None:
            simulations = self.options.simulations  # plain "go", or go mate
        return simulations, movetime

    def _tree(self, evaluator: NetworkEvaluator) -> MCTS:
        from chessnet.search.mcts import MCTS, SearchConfig

        settings = (self.options.c_puct, self.options.leaf_batch)
        if self._mcts is None or settings != self._mcts_settings:
            config = SearchConfig(c_puct=settings[0], batch_size=settings[1])
            self._mcts, self._mcts_settings = MCTS(evaluator, config), settings
        return self._mcts

    def _load(self) -> NetworkEvaluator | None:
        """The evaluator for the current Weights and Device, loading it if they changed."""
        device = resolve_device(self.options.device)
        wanted = (self.options.weights, device)
        if self._evaluator is not None and wanted == self._loaded:
            return self._evaluator
        path = Path(self.options.weights).expanduser()
        if not path.is_file():
            self.send(f"info string no weights at {path}: set the Weights option")
            return None
        from chessnet.search.evaluator import NetworkEvaluator

        try:
            evaluator = NetworkEvaluator.from_checkpoint(path, device=device, cache_size=CACHE_SIZE)
        except Exception as error:  # a bad file must not end the engine
            self.send(f"info string cannot load {path}: {type(error).__name__}: {error}")
            return None
        self._evaluator, self._loaded, self._mcts = evaluator, wanted, None
        self.send(f"info string loaded {path} on {device}")
        return evaluator


def run(lines: Iterable[str], send: Callable[[str], None], **settings: str | None) -> None:
    """Serve UCI commands from ``lines`` until ``quit`` or the end of input."""
    engine = UCIEngine(send, **settings)
    for line in lines:
        if not engine.handle(line):
            break
    engine.stop_search()


def _print_line(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def main(weights: str | None = None, device: str | None = None) -> None:
    """UCI over stdin and stdout; the entry point of ``chessnet-uci``."""
    run(sys.stdin, _print_line, weights=weights, device=device)
