"""Stub evaluators for search tests: they make the search, not a network, do the work."""

import math
from collections.abc import Callable, Sequence

import chess

from chessnet.search.evaluator import Evaluation

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def material_value(board: chess.Board) -> float:
    """``tanh(material balance / 5)`` for the side to move."""
    balance = sum(
        PIECE_VALUES[piece.piece_type] * (1 if piece.color == board.turn else -1)
        for piece in board.piece_map().values()
    )
    return math.tanh(balance / 5)


class UniformEvaluator:
    """Uniform priors over the legal moves, and a value from ``value_of`` (default 0)."""

    def __init__(self, value_of: Callable[[chess.Board], float] = lambda board: 0.0) -> None:
        self.value_of = value_of
        self.calls = 0
        self.positions = 0
        self.batch_sizes: list[int] = []

    def evaluate(self, boards: Sequence[chess.Board]) -> list[Evaluation]:
        self.calls += 1
        self.positions += len(boards)
        self.batch_sizes.append(len(boards))
        evaluations = []
        for board in boards:
            moves = list(board.legal_moves)
            priors = {move: 1 / len(moves) for move in moves}
            evaluations.append(Evaluation(priors, self.value_of(board)))
        return evaluations


class MaterialEvaluator(UniformEvaluator):
    """Uniform priors and the material balance as the value."""

    def __init__(self) -> None:
        super().__init__(material_value)


class FixedPriorEvaluator(UniformEvaluator):
    """Priors from a table of SAN moves (the rest share what is left), value 0."""

    def __init__(self, priors: dict[str, float]) -> None:
        super().__init__()
        self.san_priors = priors

    def evaluate(self, boards: Sequence[chess.Board]) -> list[Evaluation]:
        evaluations = []
        for board, evaluation in zip(boards, super().evaluate(boards), strict=True):
            moves = list(evaluation.priors)
            named = {
                move: self.san_priors[board.san(move)]
                for move in moves
                if board.san(move) in self.san_priors
            }
            rest = (1 - sum(named.values())) / max(len(moves) - len(named), 1)
            priors = {move: named.get(move, rest) for move in moves}
            evaluations.append(Evaluation(priors, evaluation.value))
        return evaluations
