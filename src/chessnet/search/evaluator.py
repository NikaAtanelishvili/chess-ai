"""Position evaluation for search: boards in, priors over the legal moves and a value out.

:class:`NetworkEvaluator` encodes boards from the side to move's perspective, runs the
network on the whole batch, takes a softmax over the legal moves only, and maps policy
indices back to real moves with :mod:`chessnet.encoding.moves`. Values are
``P(win) - P(loss)`` for the side to move. Results are cached by :func:`position_key`,
which covers everything the network sees.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import chess
import numpy as np
import numpy.typing as npt
import torch

from chessnet.encoding.board import HALFMOVE_SATURATION, encode_board
from chessnet.encoding.moves import move_to_index
from chessnet.model.resnet import ChessNet, scalar_value


class Evaluation(NamedTuple):
    """An evaluation of one position, from the side to move's perspective."""

    priors: dict[chess.Move, float]  # every legal move, summing to 1 (empty if none)
    value: float  # expected outcome in [-1, 1]


class Evaluator(Protocol):
    """Anything that evaluates positions for the search: the network, or a test stub."""

    def evaluate(self, boards: Sequence[chess.Board]) -> list[Evaluation]:
        """One evaluation per board, in order."""
        ...


def position_key(board: chess.Board) -> Hashable:
    """A key that is equal for two boards exactly when the network sees the same input.

    Pieces, side to move, castling rights, an en-passant square with a legal capture, and
    the halfmove clock up to where the network's feature saturates. The move history is not
    part of it: the network does not see it (repetitions are the search's business).
    """
    return (
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
        board.occupied_co[chess.WHITE],
        board.occupied_co[chess.BLACK],
        board.turn,
        board.clean_castling_rights(),
        board.ep_square if board.has_legal_en_passant() else None,
        min(board.halfmove_clock, HALFMOVE_SATURATION),
    )


def legal_priors(logits: npt.NDArray[np.float32], board: chess.Board) -> dict[chess.Move, float]:
    """Softmax of the policy ``logits`` (4672,) over ``board``'s legal moves only."""
    moves = list(board.legal_moves)
    if not moves:
        return {}
    indices = [move_to_index(move, board) for move in moves]
    chosen = logits[indices].astype(np.float64)
    weights = np.exp(chosen - np.max(chosen))
    weights /= weights.sum()
    return dict(zip(moves, weights.tolist(), strict=True))


class NetworkEvaluator:
    """Evaluates batches of boards with a network, with an LRU cache of recent positions.

    ``half`` (default: on for CUDA) runs the network under fp16 autocast. ``cache_size``
    bounds the cached positions; each takes a few kilobytes.
    """

    def __init__(
        self,
        network: ChessNet,
        *,
        device: str | torch.device = "cpu",
        half: bool | None = None,
        cache_size: int = 50_000,
    ) -> None:
        self.device = torch.device(device)
        self.network = network.to(self.device).eval()
        self.half = self.device.type == "cuda" if half is None else half
        self.cache_size = cache_size
        self._cache: OrderedDict[Hashable, Evaluation] = OrderedDict()
        self.positions = 0  # positions the network evaluated (cache misses)
        self.batches = 0  # network calls

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, device: str | torch.device = "cpu", **options: Any
    ) -> NetworkEvaluator:
        """An evaluator for the network in checkpoint ``path`` (any training checkpoint)."""
        from chessnet.model.checkpoint import load_network

        return cls(load_network(path, map_location=device), device=device, **options)

    def evaluate(self, boards: Sequence[chess.Board]) -> list[Evaluation]:
        keys = [position_key(board) for board in boards]
        found: dict[Hashable, Evaluation] = {}
        missing: dict[Hashable, chess.Board] = {}
        for key, board in zip(keys, boards, strict=True):
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                found[key] = cached
            elif key not in missing:
                missing[key] = board
        if missing:
            evaluations = self._run(list(missing.values()))
            for key, evaluation in zip(missing, evaluations, strict=True):
                found[key] = evaluation
                self._remember(key, evaluation)
        return [found[key] for key in keys]

    def clear_cache(self) -> None:
        self._cache.clear()

    def _remember(self, key: Hashable, evaluation: Evaluation) -> None:
        if self.cache_size <= 0:
            return
        self._cache[key] = evaluation
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _run(self, boards: Sequence[chess.Board]) -> list[Evaluation]:
        planes = torch.from_numpy(np.stack([encode_board(board) for board in boards]))
        with (
            torch.inference_mode(),
            torch.autocast(self.device.type, dtype=torch.float16, enabled=self.half),
        ):
            output = self.network(planes.to(self.device))
        logits = output.policy.float().cpu().numpy()
        values = scalar_value(output.value.float()).cpu().numpy()
        self.positions += len(boards)
        self.batches += 1
        return [
            Evaluation(legal_priors(row, board), float(value))
            for row, board, value in zip(logits, boards, values, strict=True)
        ]
