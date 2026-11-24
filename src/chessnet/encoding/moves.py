"""Move <-> policy index (AlphaZero 8x8x73 = 4672), from the side-to-move perspective.

This module is the only implementation of the policy index. Everything that turns moves
into policy indices or back goes through it.

Layout: ``index = plane * 64 + from_square``. ``from_square`` is in the mover's frame:
when Black is to move, squares are mirrored with :func:`chess.square_mirror` (a vertical
flip), the same transform :meth:`chess.Board.mirror` applies to the board. ``plane`` is
one of 73 move types, with ranks and files counted in the mover's frame:

* 0-55, queen-like moves: ``plane = 7 * direction + (distance - 1)`` for the directions
  N, NE, E, SE, S, SW, W, NW (N = towards the opponent) and distances 1-7. Queen
  promotions and castling (the king's two-square move) use these planes.
* 56-63, knight moves, clockwise from (+2 ranks, +1 file): (+2,+1), (+1,+2), (-1,+2),
  (-2,+1), (-2,-1), (-1,-2), (+1,-2), (+2,-1).
* 64-72, underpromotions: ``plane = 64 + 3 * direction + piece`` for the directions
  "capture towards the a-file", "straight", "capture towards the h-file" and the pieces
  knight, bishop, rook.

The flat order matches a convolutional policy head of shape ``(73, 8, 8)`` flattened
row-major, given that ``square == 8 * rank + file``.
"""

from __future__ import annotations

from typing import Final

import chess
import numpy as np
import numpy.typing as npt

NUM_MOVE_TYPES: Final = 73
POLICY_SIZE: Final = NUM_MOVE_TYPES * 64  # 4672

#: (rank step, file step) of the 8 queen-like directions, in plane order.
QUEEN_DIRECTIONS: Final = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
MAX_DISTANCE: Final = 7
KNIGHT_PLANE_OFFSET: Final = len(QUEEN_DIRECTIONS) * MAX_DISTANCE  # 56
#: (rank step, file step) of the 8 knight jumps, in plane order.
KNIGHT_JUMPS: Final = ((2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1))
UNDERPROMOTION_PLANE_OFFSET: Final = KNIGHT_PLANE_OFFSET + len(KNIGHT_JUMPS)  # 64
#: File steps of the underpromotion directions, in plane order (the rank step is +1).
UNDERPROMOTION_FILE_STEPS: Final = (-1, 0, 1)
#: Underpromotion pieces, in plane order.
UNDERPROMOTION_PIECES: Final = (chess.KNIGHT, chess.BISHOP, chess.ROOK)

#: Number of indices that decode to an on-board move (as in Leela Chess Zero's 1858).
NUM_VALID_INDICES: Final = 1858

_PROMOTION_FROM_RANK: Final = 6  # the 7th rank, in the mover's frame
_LAST_RANK: Final = 7


def _plane_steps() -> tuple[tuple[int, int], ...]:
    """(rank step, file step) of every plane."""
    steps: list[tuple[int, int]] = []
    for rank_step, file_step in QUEEN_DIRECTIONS:
        steps.extend(
            (rank_step * distance, file_step * distance) for distance in range(1, MAX_DISTANCE + 1)
        )
    steps.extend(KNIGHT_JUMPS)
    for file_step in UNDERPROMOTION_FILE_STEPS:
        steps.extend((1, file_step) for _ in UNDERPROMOTION_PIECES)
    assert len(steps) == NUM_MOVE_TYPES
    return tuple(steps)


def _build_tables() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Precompute both directions of the index.

    Returns ``(plane_by_from_to, destination)``: the queen-like or knight plane of each
    ``from_square * 64 + to_square`` pair (-1 if none), and the destination square of each
    policy index (-1 if the index does not encode an on-board move).
    """
    plane_by_from_to = [-1] * (64 * 64)
    destination = [-1] * POLICY_SIZE
    for plane, (rank_step, file_step) in enumerate(_plane_steps()):
        is_underpromotion = plane >= UNDERPROMOTION_PLANE_OFFSET
        for from_square in chess.SQUARES:
            from_rank = chess.square_rank(from_square)
            rank = from_rank + rank_step
            file = chess.square_file(from_square) + file_step
            off_board = not (0 <= rank < 8 and 0 <= file < 8)
            if off_board or (is_underpromotion and from_rank != _PROMOTION_FROM_RANK):
                continue
            to_square = chess.square(file, rank)
            destination[plane * 64 + from_square] = to_square
            if not is_underpromotion:
                plane_by_from_to[from_square * 64 + to_square] = plane
    return tuple(plane_by_from_to), tuple(destination)


_PLANE_BY_FROM_TO, _DESTINATION = _build_tables()

_UNDERPROMOTION_PLANES: Final = {
    (file_step, piece): UNDERPROMOTION_PLANE_OFFSET + 3 * direction + piece_index
    for direction, file_step in enumerate(UNDERPROMOTION_FILE_STEPS)
    for piece_index, piece in enumerate(UNDERPROMOTION_PIECES)
}


def _is_promotion_step(from_square: chess.Square, to_square: chess.Square) -> bool:
    """Whether a move (in the mover's frame) steps from the 7th rank onto the last rank."""
    return (
        chess.square_rank(from_square) == _PROMOTION_FROM_RANK
        and chess.square_rank(to_square) == _LAST_RANK
        and abs(chess.square_file(to_square) - chess.square_file(from_square)) <= 1
    )


def mirror_move(move: chess.Move) -> chess.Move:
    """Flip a move vertically (``chess.square_mirror`` on both squares)."""
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
        drop=move.drop,
    )


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Policy index of ``move`` played in ``board`` (from the side to move's view).

    Only the move's geometry and promotion piece are encoded; legality is not checked.

    Raises:
        ValueError: if the move cannot be represented (null moves, drops, promotions
            to a king or pawn, promotions that are not a one-step pawn move onto the last
            rank, or steps that are neither queen-like nor knight moves).
    """
    if move.drop is not None:
        raise ValueError(f"drops are not representable: {move.uci()}")
    from_square, to_square = move.from_square, move.to_square
    if board.turn == chess.BLACK:
        from_square = chess.square_mirror(from_square)
        to_square = chess.square_mirror(to_square)

    if move.promotion is not None:
        if not _is_promotion_step(from_square, to_square):
            raise ValueError(f"promotion is not a step onto the last rank: {move.uci()}")
        if move.promotion != chess.QUEEN:
            file_step = chess.square_file(to_square) - chess.square_file(from_square)
            plane = _UNDERPROMOTION_PLANES.get((file_step, move.promotion))
            if plane is None:
                raise ValueError(f"promotion piece is not representable: {move.uci()}")
            return plane * 64 + from_square

    plane = _PLANE_BY_FROM_TO[from_square * 64 + to_square]
    if plane < 0:
        raise ValueError(f"move is not representable: {move.uci()}")
    return plane * 64 + from_square


def _mover_has_pawn_on(board: chess.Board, square: chess.Square) -> bool:
    """Whether the side to move has a pawn on ``square`` (given in the mover's frame)."""
    if board.turn == chess.BLACK:
        square = chess.square_mirror(square)
    return bool(board.pawns & board.occupied_co[board.turn] & chess.BB_SQUARES[square])


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Inverse of :func:`move_to_index`: the real (un-mirrored) move for ``board``.

    Queen-like planes that move one of the mover's pawns onto the last rank decode to a
    queen promotion. Legality is not checked.

    Raises:
        ValueError: if ``index`` is out of range or does not encode an on-board move.
    """
    if not 0 <= index < POLICY_SIZE:
        raise ValueError(f"policy index out of range: {index}")
    to_square = _DESTINATION[index]
    if to_square < 0:
        raise ValueError(f"policy index does not encode an on-board move: {index}")
    plane, from_square = divmod(index, 64)

    promotion: chess.PieceType | None = None
    if plane >= UNDERPROMOTION_PLANE_OFFSET:
        promotion = UNDERPROMOTION_PIECES[(plane - UNDERPROMOTION_PLANE_OFFSET) % 3]
    elif _is_promotion_step(from_square, to_square) and _mover_has_pawn_on(board, from_square):
        promotion = chess.QUEEN

    if board.turn == chess.BLACK:
        from_square = chess.square_mirror(from_square)
        to_square = chess.square_mirror(to_square)
    return chess.Move(from_square, to_square, promotion=promotion)


def legal_mask(board: chess.Board) -> npt.NDArray[np.bool_]:
    """Boolean mask of shape ``(4672,)`` that is true exactly at the legal moves' indices."""
    mask = np.zeros(POLICY_SIZE, dtype=np.bool_)
    for move in board.legal_moves:
        mask[move_to_index(move, board)] = True
    return mask
