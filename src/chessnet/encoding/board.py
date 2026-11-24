"""Board to input planes, from the side-to-move perspective.

Conventions (documented in ``docs/ARCHITECTURE.md``):

* If Black is to move, the board is replaced by ``board.mirror()``: flipped vertically,
  with piece colours, castling rights, the en-passant square and the turn swapped. After
  that the side to move is always "White".
* Planes are indexed ``planes[plane, rank, file]``, so ``planes[:, 0, 0]`` is a1 from the
  mover's perspective. :func:`square_to_rank_file` owns this mapping; vectorised code uses
  :data:`CELL_SQUARES`, which is derived from it.
* Layout is channels-first (NCHW once batched), float32.

Planes (``ENCODING_VERSION`` 1):

====== ===============================================================
0-5    mover's pawns, knights, bishops, rooks, queens, king
6-11   opponent's pawns, knights, bishops, rooks, queens, king
12-15  constant: mover O-O, mover O-O-O, opponent O-O, opponent O-O-O
16     en-passant target square, only if a legal en-passant capture exists
17     constant: halfmove clock / 100, clipped to 1
18     constant: all ones
====== ===============================================================
"""

from __future__ import annotations

from typing import Final

import chess
import numpy as np
import numpy.typing as npt

NUM_PLANES: Final = 19
PLANE_SHAPE: Final = (NUM_PLANES, 8, 8)

#: Piece order within the mover's (0-5) and the opponent's (6-11) piece planes.
PIECE_TYPES: Final = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)
MOVER_PIECE_PLANE: Final = 0
OPPONENT_PIECE_PLANE: Final = 6
CASTLING_PLANE: Final = 12
EN_PASSANT_PLANE: Final = 16
HALFMOVE_PLANE: Final = 17
ONES_PLANE: Final = 18

#: The halfmove-clock feature saturates at this many plies (the 50-move rule).
HALFMOVE_SATURATION: Final = 100
#: Largest halfmove clock that is stored; larger clocks are clipped (same feature value).
HALFMOVE_CLOCK_MAX: Final = 255


def _halfmove_feature_table() -> npt.NDArray[np.float32]:
    clocks = np.minimum(np.arange(HALFMOVE_CLOCK_MAX + 1), HALFMOVE_SATURATION)
    table = (clocks / HALFMOVE_SATURATION).astype(np.float32)
    table.setflags(write=False)
    return table


#: ``HALFMOVE_FEATURE[min(clock, HALFMOVE_CLOCK_MAX)]`` is the value of plane 17. Every
#: encoder reads this one float32 table, so CPU and GPU paths agree bit for bit.
HALFMOVE_FEATURE: Final = _halfmove_feature_table()


def square_to_rank_file(square: chess.Square) -> tuple[int, int]:
    """Return the ``(rank, file)`` cell of ``square`` in a plane (a1 is ``(0, 0)``)."""
    return chess.square_rank(square), chess.square_file(square)


#: The square shown by each cell of a plane flattened in row-major ``[rank, file]`` order.
CELL_SQUARES: Final[tuple[chess.Square, ...]] = tuple(
    sorted(chess.SQUARES, key=square_to_rank_file)
)


def perspective_board(board: chess.Board) -> chess.Board:
    """Return ``board`` as seen by the side to move, which is White afterwards.

    Black-to-move boards are mirrored with :meth:`chess.Board.mirror` (a new board without
    move stack). White-to-move boards are returned as they are, not copied, so the result
    must not be modified.
    """
    return board.mirror() if board.turn == chess.BLACK else board


def castling_flags(view: chess.Board) -> tuple[bool, bool, bool, bool]:
    """Castling rights of a perspective board.

    Returns ``(mover O-O, mover O-O-O, opponent O-O, opponent O-O-O)``.
    """
    return (
        view.has_kingside_castling_rights(chess.WHITE),
        view.has_queenside_castling_rights(chess.WHITE),
        view.has_kingside_castling_rights(chess.BLACK),
        view.has_queenside_castling_rights(chess.BLACK),
    )


def en_passant_square(view: chess.Board) -> chess.Square | None:
    """En-passant target square of a perspective board, if a legal capture exists.

    python-chess keeps a potential en-passant square after every double pawn push. Only
    squares where an en-passant capture is actually legal are encoded, so that positions
    which are identical for repetition purposes also encode identically.
    """
    return view.ep_square if view.has_legal_en_passant() else None


def encode_board(board: chess.Board) -> npt.NDArray[np.float32]:
    """Encode ``board`` as ``(19, 8, 8)`` float32 planes from the side to move's view."""
    view = perspective_board(board)
    planes = np.zeros(PLANE_SHAPE, dtype=np.float32)

    for square, piece in view.piece_map().items():
        rank, file = square_to_rank_file(square)
        first = MOVER_PIECE_PLANE if piece.color == chess.WHITE else OPPONENT_PIECE_PLANE
        planes[first + piece.piece_type - chess.PAWN, rank, file] = 1.0

    for offset, has_right in enumerate(castling_flags(view)):
        if has_right:
            planes[CASTLING_PLANE + offset] = 1.0

    ep_square = en_passant_square(view)
    if ep_square is not None:
        rank, file = square_to_rank_file(ep_square)
        planes[EN_PASSANT_PLANE, rank, file] = 1.0

    planes[HALFMOVE_PLANE] = HALFMOVE_FEATURE[min(view.halfmove_clock, HALFMOVE_CLOCK_MAX)]
    planes[ONES_PLANE] = 1.0
    return planes
