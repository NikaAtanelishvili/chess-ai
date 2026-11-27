"""Players: random, policy-only (plain or with safety rules) and MCTS behind one interface."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Protocol

import chess

from chessnet.search.evaluator import Evaluator
from chessnet.search.mcts import MCTS, SearchConfig, SearchResult, sample_move


class Player(Protocol):
    """Chooses a legal move in a position (the board keeps the game's move stack)."""

    def choose(self, board: chess.Board) -> chess.Move:
        """A legal move for the side to move in ``board``."""
        ...


def _require_moves(board: chess.Board) -> list[chess.Move]:
    moves = list(board.legal_moves)
    if not moves:
        raise ValueError("no legal moves: the game is over")
    return moves


class RandomPlayer:
    """A uniformly random legal move: the weakest baseline, for tests and evaluation."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose(self, board: chess.Board) -> chess.Move:
        return self._rng.choice(_require_moves(board))


class PolicyPlayer:
    """The network's most likely legal move, without search: fast, and the main baseline."""

    def __init__(self, evaluator: Evaluator) -> None:
        self.evaluator = evaluator

    def choose(self, board: chess.Board) -> chess.Move:
        _require_moves(board)
        priors = self.evaluator.evaluate([board])[0].priors
        return max(priors, key=priors.__getitem__)


def safe_policy_move(
    board: chess.Board, priors: Mapping[chess.Move, float], *, value: float
) -> chess.Move:
    """The most likely move after these rules, in prior order:

    1. a move that checkmates, if there is one;
    2. when the side to move is better (``value``, the network's expected outcome for it,
       above 0), the first move that neither stalemates nor leaves a threefold repetition to
       claim (completed by the move, or reachable by the opponent's reply), since a draw
       would give the advantage away;
    3. otherwise the most likely move: level or worse, a draw is welcome. Also when every
       move would draw.

    ``board`` is restored before returning.
    """
    ranked = sorted(priors, key=priors.__getitem__, reverse=True)
    for move in ranked:
        board.push(move)
        mates = board.is_checkmate()
        board.pop()
        if mates:
            return move
    if value <= 0.0:
        return ranked[0]
    for move in ranked:
        board.push(move)
        unsafe = board.is_stalemate() or board.can_claim_threefold_repetition()
        board.pop()
        if not unsafe:
            return move
    return ranked[0]


class PolicySafePlayer:
    """The policy with safety rules (no search): it takes a mate in one, and when the value
    head rates its position better than level it avoids moves that stalemate or allow a
    threefold-repetition claim while it has others. See :func:`safe_policy_move`."""

    def __init__(self, evaluator: Evaluator) -> None:
        self.evaluator = evaluator

    def choose(self, board: chess.Board) -> chess.Move:
        _require_moves(board)
        [evaluation] = self.evaluator.evaluate([board])
        return safe_policy_move(board, evaluation.priors, value=evaluation.value)


class MCTSPlayer:
    """PUCT search with a playout or time budget per move, reusing its tree between moves.

    With ``temperature`` above 0, the first ``temperature_plies`` plies of a game are drawn
    from the visit counts (for variety in evaluation matches); otherwise it plays the most
    visited move. ``last_result`` holds the latest search's statistics.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        *,
        simulations: int | None = None,
        movetime: float | None = None,
        config: SearchConfig | None = None,
        temperature: float = 0.0,
        temperature_plies: int = 0,
        seed: int | None = None,
    ) -> None:
        if simulations is None and movetime is None:
            raise ValueError("an MCTS player needs simulations, movetime or both")
        self.mcts = MCTS(evaluator, config)
        self.simulations = simulations
        self.movetime = movetime
        self.temperature = temperature
        self.temperature_plies = temperature_plies
        self._rng = random.Random(seed)
        self.last_result: SearchResult | None = None

    def choose(self, board: chess.Board) -> chess.Move:
        result = self.mcts.search(board, simulations=self.simulations, movetime=self.movetime)
        self.last_result = result
        if self.temperature > 0 and board.ply() < self.temperature_plies:
            return sample_move(result.visits, self.temperature, self._rng)
        return result.move
