import chess
import numpy as np
import pytest

from chessnet.encoding import ENCODING_VERSION
from chessnet.encoding.board import (
    CASTLING_PLANE,
    CELL_SQUARES,
    EN_PASSANT_PLANE,
    HALFMOVE_FEATURE,
    HALFMOVE_PLANE,
    MOVER_PIECE_PLANE,
    NUM_PLANES,
    ONES_PLANE,
    OPPONENT_PIECE_PLANE,
    PLANE_SHAPE,
    encode_board,
    perspective_board,
    square_to_rank_file,
)

from helpers import positions

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(6)  # offsets within a side's piece planes


def board_after(*sans: str) -> chess.Board:
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


def test_encoding_version() -> None:
    assert ENCODING_VERSION == 1


def test_shape_and_dtype() -> None:
    planes = encode_board(chess.Board())
    assert planes.shape == PLANE_SHAPE == (NUM_PLANES, 8, 8) == (19, 8, 8)
    assert planes.dtype == np.float32
    assert planes.flags.c_contiguous


@pytest.mark.parametrize(
    ("square", "cell"),
    [(chess.A1, (0, 0)), (chess.H1, (0, 7)), (chess.A8, (7, 0)), (chess.E2, (1, 4))],
)
def test_square_to_rank_file(square: chess.Square, cell: tuple[int, int]) -> None:
    assert square_to_rank_file(square) == cell


def test_cell_squares_follow_rank_major_order() -> None:
    assert list(CELL_SQUARES) == list(range(64))
    for index, square in enumerate(CELL_SQUARES):
        assert square_to_rank_file(square) == divmod(index, 8)


def test_orientation_white_pawn_on_e2() -> None:
    planes = encode_board(chess.Board())
    assert planes[MOVER_PIECE_PLANE + PAWN, 1, 4] == 1.0


def test_orientation_black_pawn_on_e7_after_mirroring() -> None:
    board = board_after("Nf3")  # Black to move, e7 pawn untouched
    planes = encode_board(board)
    assert planes[MOVER_PIECE_PLANE + PAWN, 1, 4] == 1.0  # Black's e7 pawn, seen from Black
    assert planes[OPPONENT_PIECE_PLANE + PAWN, 6, 4] == 1.0  # White's e2 pawn
    assert planes[OPPONENT_PIECE_PLANE + KNIGHT, 5, 5] == 1.0  # White's Nf3 appears on f6


def test_start_position_piece_planes() -> None:
    planes = encode_board(chess.Board())
    expected = np.zeros((12, 8, 8), dtype=np.float32)
    back_rank = (ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK)
    for file, piece in enumerate(back_rank):
        expected[MOVER_PIECE_PLANE + piece, 0, file] = 1.0
        expected[OPPONENT_PIECE_PLANE + piece, 7, file] = 1.0
    expected[MOVER_PIECE_PLANE + PAWN, 1, :] = 1.0
    expected[OPPONENT_PIECE_PLANE + PAWN, 6, :] = 1.0
    np.testing.assert_array_equal(planes[:12], expected)


@pytest.mark.parametrize(
    ("fen", "expected"),
    [
        # White to move: mover = White (K only), opponent = Black (q only).
        ("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1", (1, 0, 0, 1)),
        # Black to move: mover = Black (q only), opponent = White (K only).
        ("r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1", (0, 1, 1, 0)),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", (1, 1, 1, 1)),
        ("r3k2r/8/8/8/8/8/8/R3K2R b - - 0 1", (0, 0, 0, 0)),
    ],
)
def test_castling_planes(fen: str, expected: tuple[int, int, int, int]) -> None:
    planes = encode_board(chess.Board(fen))
    for offset, value in enumerate(expected):
        assert np.all(planes[CASTLING_PLANE + offset] == value)


@pytest.mark.parametrize("fen", [positions.EN_PASSANT_WHITE, positions.EN_PASSANT_BLACK])
def test_en_passant_plane_marks_target_square_for_both_colours(fen: str) -> None:
    planes = encode_board(chess.Board(fen))
    expected = np.zeros((8, 8), dtype=np.float32)
    expected[5, 5] = 1.0  # f6 for White; Black's f3 mirrors to f6
    np.testing.assert_array_equal(planes[EN_PASSANT_PLANE], expected)


def test_en_passant_plane_empty_without_legal_capture() -> None:
    board = board_after("e4")  # python-chess sets ep_square = e3, but no capture exists
    assert board.ep_square is not None
    assert not encode_board(board)[EN_PASSANT_PLANE].any()


@pytest.mark.parametrize("clock", [0, 1, 37, 99, 100, 101, 255, 256, 1000])
def test_halfmove_plane(clock: int) -> None:
    board = chess.Board()
    board.halfmove_clock = clock
    plane = encode_board(board)[HALFMOVE_PLANE]
    assert np.all(plane == np.float32(min(clock, 100) / 100))


def test_halfmove_feature_table_is_read_only() -> None:
    assert HALFMOVE_FEATURE.shape == (256,)
    assert not HALFMOVE_FEATURE.flags.writeable


def test_ones_plane() -> None:
    assert np.all(encode_board(chess.Board())[ONES_PLANE] == 1.0)


def test_perspective_board_keeps_white_and_mirrors_black() -> None:
    white = chess.Board()
    assert perspective_board(white) is white

    black = board_after("e4")
    view = perspective_board(black)
    assert view is not black
    assert view.turn == chess.WHITE
    assert view.piece_at(chess.E5) == chess.Piece(chess.PAWN, chess.BLACK)  # White's e4 pawn


@pytest.mark.parametrize("fen", positions.ALL_FIXTURES)
def test_fixtures_are_valid_positions(fen: str) -> None:
    assert chess.Board(fen).is_valid()


def test_perspective_board_does_not_copy_the_move_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    # Board.mirror() copies the whole move stack before discarding it, which made encoding
    # cost grow with the length of the game.
    board = chess.Board()
    for _ in range(41):
        board.push(next(iter(board.legal_moves)))
    assert board.turn == chess.BLACK
    stack_arguments: list[object] = []
    original_copy = chess.Board.copy

    def spy(self: chess.Board, *, stack: bool | int = True) -> chess.Board:
        stack_arguments.append(stack)
        return original_copy(self, stack=stack)

    monkeypatch.setattr(chess.Board, "copy", spy)
    view = perspective_board(board)
    assert stack_arguments == [False]
    assert view.move_stack == []
    assert view.fen() == board.mirror().fen()
