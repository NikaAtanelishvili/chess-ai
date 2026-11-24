"""Corner squares through encode -> pack -> unpack on the CPU path, including bit 63 (h8).

In torch, bitboards are int64, so a piece on h8 makes the value negative (exactly -2**63 when
it is alone). These tests fail if any code decides "bit set" with a sign-dependent test such
as ``> 0`` instead of extracting bits.
"""

import chess
import numpy as np
import pytest

from chessnet.encoding.board import (
    MOVER_PIECE_PLANE,
    OPPONENT_PIECE_PLANE,
    encode_board,
    perspective_board,
)
from chessnet.encoding.packed import (
    batch_from_records,
    empty_records,
    pack_position,
    unpack_batch_torch,
)
from chessnet.encoding.value import DRAW

# All four corners are occupied in every fixture. Third column: who owns the piece on h8 in
# the mover's frame; fourth: its type. The Black-to-move fixtures mirror the White ones.
CORNER_FIXTURES = [
    # Mover's knight h8, mover's rook a1, opponent's knight a8 and bishop h1.
    ("n6N/8/4k3/8/8/4K3/8/R6b w - - 0 1", "mover", chess.KNIGHT),
    # Opponent's rook h8 and knight a1, mover's bishop a8 and rook h1.
    ("B6r/8/4k3/8/8/4K3/8/n6R w - - 0 1", "opponent", chess.ROOK),
    # Black to move: Black's knight on h1 becomes the mover's knight on h8.
    ("r6B/8/4k3/8/8/4K3/8/N6n b - - 0 1", "mover", chess.KNIGHT),
    # Black to move: White's rook on h1 becomes the opponent's rook on h8.
    ("N6r/8/4k3/8/8/4K3/8/b6R b - - 0 1", "opponent", chess.ROOK),
]
CORNER_CELLS = {chess.A1: (0, 0), chess.H1: (0, 7), chess.A8: (7, 0), chess.H8: (7, 7)}


def piece_plane(piece: chess.Piece) -> int:
    """Plane of a piece on a perspective board (the mover is White there)."""
    first = MOVER_PIECE_PLANE if piece.color == chess.WHITE else OPPONENT_PIECE_PLANE
    return first + piece.piece_type - chess.PAWN


@pytest.mark.parametrize(("fen", "owner", "piece_type"), CORNER_FIXTURES)
def test_corner_pieces_survive_encode_pack_unpack(
    fen: str, owner: str, piece_type: chess.PieceType
) -> None:
    board = chess.Board(fen)
    assert board.is_valid()
    view = perspective_board(board)
    on_h8 = view.piece_at(chess.H8)
    assert on_h8 == chess.Piece(piece_type, chess.WHITE if owner == "mover" else chess.BLACK)

    batch = batch_from_records(pack_position(board, next(iter(board.legal_moves)), DRAW))
    assert batch.bitboards[0, piece_plane(on_h8)].item() == -(2**63)  # only bit 63 set

    planes = unpack_batch_torch(batch)[0].numpy()
    assert planes.tobytes() == encode_board(board).tobytes()
    for square, (rank, file) in CORNER_CELLS.items():
        piece = view.piece_at(square)
        assert piece is not None
        pieces_on_cell = planes[:12, rank, file]
        assert pieces_on_cell.sum() == 1.0
        assert pieces_on_cell[piece_plane(piece)] == 1.0


@pytest.mark.parametrize(
    ("bitboard", "squares"),
    [
        (1 << 63, [chess.H8]),
        (1, [chess.A1]),
        (1 << 7, [chess.H1]),
        (1 << 56, [chess.A8]),
        ((1 << 63) | 1, [chess.A1, chess.H8]),
        (2**64 - 1, list(chess.SQUARES)),
    ],
)
def test_extreme_bitboards_unpack_bit_for_bit(bitboard: int, squares: list[chess.Square]) -> None:
    records = empty_records(2)
    records["ep"] = -1
    records["bb"][0, 0] = bitboard  # first plane (mover's pawns)
    records["bb"][1, 11] = bitboard  # last piece plane (opponent's king)
    planes = unpack_batch_torch(batch_from_records(records)).numpy()

    expected = np.zeros((8, 8), dtype=np.float32)
    for square in squares:
        expected[chess.square_rank(square), chess.square_file(square)] = 1.0
    np.testing.assert_array_equal(planes[0, 0], expected)
    np.testing.assert_array_equal(planes[1, 11], expected)
    assert planes[0, 1:12].sum() == 0
    assert planes[1, :11].sum() == 0
