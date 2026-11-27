"""Players for evaluation, described by specs so that worker processes can rebuild them.

A :class:`PlayerSpec` names a random mover, the network's policy, an MCTS player, or an
external UCI engine such as Stockfish. ``chessnet.eval.workers`` runs games and puzzles with
the players built once per worker process.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import chess
import chess.engine

from chessnet.engine.player import MCTSPlayer, Player, PolicyPlayer, PolicySafePlayer, RandomPlayer

OptionValue = str | int | bool


@dataclass(frozen=True)
class PlayerSpec:
    """How to build a player (kind: random, policy, policy-safe, mcts or uci)."""

    name: str
    kind: str
    weights: str | None = None
    simulations: int | None = None  # mcts: playouts per move
    movetime: float | None = None  # mcts or uci: seconds per move
    nodes: int | None = None  # uci: nodes per move
    leaf_batch: int = 32
    command: tuple[str, ...] = ()  # uci: the engine's command line
    options: tuple[tuple[str, OptionValue], ...] = ()  # uci: options to set
    seed: int | None = None  # random
    timeout: float = 30.0  # uci: seconds beyond the move time before the engine counts as stuck

    def describe(self) -> dict[str, Any]:
        """The spec without empty fields, for reports."""
        fields = asdict(self)
        fields["options"] = dict(self.options)
        return {key: value for key, value in fields.items() if value not in (None, (), {})}


class UCIPlayer:
    """An external UCI engine as a player, with a fixed limit per move."""

    def __init__(
        self,
        command: tuple[str, ...],
        options: dict[str, OptionValue],
        *,
        nodes: int | None = None,
        movetime: float | None = None,
        timeout: float = 30.0,
    ) -> None:
        if nodes is None and movetime is None:
            raise ValueError("a UCI player needs nodes or movetime per move")
        # python-chess gives up on a call after ``timeout`` seconds (plus the move time).
        self.engine = chess.engine.SimpleEngine.popen_uci(list(command), timeout=timeout)
        if options:
            self.engine.configure(options)
        self.limit = chess.engine.Limit(nodes=nodes, time=movetime)

    def choose(self, board: chess.Board) -> chess.Move:
        move = self.engine.play(board, self.limit).move
        if move is None:
            raise ValueError("the engine returned no move")
        return move

    def close(self) -> None:
        """Quit the engine; kill it if it does not quit in time. Safe to call twice."""
        try:
            self.engine.quit()
        except Exception:
            with contextlib.suppress(Exception):
                self.engine.close()


def uci_option_range(command: tuple[str, ...], name: str) -> tuple[int, int]:
    """The (min, max) an engine announces for a spin option, e.g. Stockfish's UCI_Elo."""
    engine = chess.engine.SimpleEngine.popen_uci(list(command), timeout=60)
    try:
        option = engine.options[name]
        if option.min is None or option.max is None:
            raise ValueError(f"{name} is not a spin option")
        return int(option.min), int(option.max)
    finally:
        engine.quit()


def resolve_device(device: str, worker: int = 0) -> str:
    """``auto`` becomes cuda or cpu; workers are spread over the GPUs round-robin."""
    if device == "cpu":
        return "cpu"
    import torch

    if not torch.cuda.is_available():
        if device == "cuda":
            raise ValueError("CUDA was requested but is not available")
        return "cpu"
    return f"cuda:{worker % torch.cuda.device_count()}"


def build_players(specs: tuple[PlayerSpec, ...], device: str) -> tuple[Player, ...]:
    """Players for ``specs``; players with the same weights share one evaluator (and cache)."""
    from chessnet.search.evaluator import NetworkEvaluator
    from chessnet.search.mcts import SearchConfig

    evaluators: dict[str, NetworkEvaluator] = {}
    players: list[Player] = []
    for spec in specs:
        if spec.kind == "random":
            players.append(RandomPlayer(seed=spec.seed))
        elif spec.kind == "uci":
            options = dict(spec.options)
            players.append(
                UCIPlayer(
                    spec.command,
                    options,
                    nodes=spec.nodes,
                    movetime=spec.movetime,
                    timeout=spec.timeout,
                )
            )
        elif spec.kind in ("policy", "policy-safe", "mcts"):
            if spec.weights is None:
                raise ValueError(f"{spec.name} needs weights")
            if spec.weights not in evaluators:
                evaluators[spec.weights] = NetworkEvaluator.from_checkpoint(
                    spec.weights, device=device, cache_size=20_000
                )
            evaluator = evaluators[spec.weights]
            if spec.kind == "policy":
                players.append(PolicyPlayer(evaluator))
            elif spec.kind == "policy-safe":
                players.append(PolicySafePlayer(evaluator))
            else:
                players.append(
                    MCTSPlayer(
                        evaluator,
                        simulations=spec.simulations,
                        movetime=spec.movetime,
                        config=SearchConfig(batch_size=spec.leaf_batch),
                    )
                )
        else:
            raise ValueError(f"unknown player kind {spec.kind!r}")
    return tuple(players)


def close_players(players: Iterable[Player]) -> None:
    for player in players:
        close = getattr(player, "close", None)
        if close is not None:
            close()
