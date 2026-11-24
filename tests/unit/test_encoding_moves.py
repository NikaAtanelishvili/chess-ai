import chess
import numpy as np
import pytest

from chessnet.encoding.moves import (
    NUM_VALID_INDICES,
    POLICY_SIZE,
    index_to_move,
    legal_mask,
    mirror_move,
    move_to_index,
)

from helpers import positions

AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

# Golden indices: they pin the documented layout (plane * 64 + from_square). Changing any
# of them changes the encoding and requires an ENCODING_VERSION bump.
GOLDEN = [
    (positions.START, "e2e4", 76),  # N, distance 2, from e2
    (AFTER_E4, "e7e5", 76),  # Black's mirror image of e2e4
    (positions.START, "g1f3", 4038),  # knight jump (+2, -1), from g1
    (AFTER_E4, "g8f6", 4038),
    (positions.CASTLING_WHITE, "e1g1", 964),  # O-O: E, distance 2, from e1
    (positions.CASTLING_BLACK, "e8g8", 964),
    (positions.CASTLING_WHITE, "e1c1", 2756),  # O-O-O: W, distance 2, from e1
    (positions.CASTLING_BLACK, "e8c8", 2756),
    (positions.EN_PASSANT_WHITE, "e5f6", 484),  # NE, distance 1, from e5
    (positions.EN_PASSANT_BLACK, "e4f3", 484),
]

# White promotions from b7 (plain: b8; captures: a8 and c8). Black's promotions from b2
# are their mirror images and get the same indices.
WHITE_PROMOTIONS = {
    "b7b8q": 49,
    "b7a8q": 3185,
    "b7c8q": 497,
    "b7a8n": 4145,
    "b7a8b": 4209,
    "b7a8r": 4273,
    "b7b8n": 4337,
    "b7b8b": 4401,
    "b7b8r": 4465,
    "b7c8n": 4529,
    "b7c8b": 4593,
    "b7c8r": 4657,
}


def black_equivalent(uci: str) -> str:
    return uci.translate(str.maketrans("78", "21"))


PROMOTIONS = [(positions.PROMOTION_WHITE, uci, i) for uci, i in WHITE_PROMOTIONS.items()] + [
    (positions.PROMOTION_BLACK, black_equivalent(uci), i) for uci, i in WHITE_PROMOTIONS.items()
]


@pytest.mark.parametrize(("fen", "uci", "index"), GOLDEN + PROMOTIONS)
def test_specific_moves_have_documented_indices(fen: str, uci: str, index: int) -> None:
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves
    assert move_to_index(move, board) == index
    assert index_to_move(index, board) == move


def test_promotion_fixtures_cover_all_pieces_with_and_without_capture() -> None:
    for fen in (positions.PROMOTION_WHITE, positions.PROMOTION_BLACK):
        board = chess.Board(fen)
        promotions = [m for m in board.legal_moves if m.promotion]
        captures = {m.promotion for m in promotions if board.is_capture(m)}
        quiet = {m.promotion for m in promotions if not board.is_capture(m)}
        all_pieces = {chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT}
        assert captures == quiet == all_pieces


@pytest.mark.parametrize("fen", positions.ALL_FIXTURES)
def test_fixture_moves_round_trip_with_unique_indices(fen: str) -> None:
    board = chess.Board(fen)
    indices = [move_to_index(move, board) for move in board.legal_moves]
    assert len(set(indices)) == len(indices)
    for move, index in zip(board.legal_moves, indices, strict=True):
        assert index_to_move(index, board) == move


@pytest.mark.parametrize("fen", positions.ALL_FIXTURES)
def test_legal_mask_is_exactly_the_legal_moves(fen: str) -> None:
    board = chess.Board(fen)
    mask = legal_mask(board)
    assert mask.shape == (POLICY_SIZE,)
    assert mask.dtype == np.bool_
    assert mask.sum() == board.legal_moves.count()
    expected = {move_to_index(move, board) for move in board.legal_moves}
    assert set(np.flatnonzero(mask).tolist()) == expected


def test_max_mobility_position() -> None:
    board = chess.Board(positions.MAX_MOBILITY)
    assert board.legal_moves.count() == positions.MAX_MOBILITY_MOVES
    assert legal_mask(board).sum() == positions.MAX_MOBILITY_MOVES


@pytest.mark.parametrize("turn", [chess.WHITE, chess.BLACK])
def test_every_on_board_index_round_trips(turn: chess.Color) -> None:
    board = chess.Board.empty()
    board.turn = turn
    valid = 0
    for index in range(POLICY_SIZE):
        try:
            move = index_to_move(index, board)
        except ValueError:
            continue
        valid += 1
        assert move.promotion is None or move.promotion != chess.QUEEN
        assert move_to_index(move, board) == index
    assert valid == NUM_VALID_INDICES


@pytest.mark.parametrize(
    ("fen", "turn"),
    [
        ("8/PPPPPPPP/8/8/8/8/8/8 w - - 0 1", chess.WHITE),
        ("8/8/8/8/8/8/pppppppp/8 b - - 0 1", chess.BLACK),
    ],
)
def test_pawn_steps_onto_last_rank_decode_to_queen_promotions(fen: str, turn: chess.Color) -> None:
    board = chess.Board(fen)
    assert board.turn == turn
    queen_promotions = 0
    for index in range(POLICY_SIZE):
        try:
            move = index_to_move(index, board)
        except ValueError:
            continue
        queen_promotions += move.promotion == chess.QUEEN
        assert move_to_index(move, board) == index
    assert queen_promotions == 8 + 7 + 7  # straight, capture left, capture right


def test_non_pawn_step_onto_last_rank_is_not_a_promotion() -> None:
    board = chess.Board("4k3/R7/8/8/8/8/8/4K3 w - - 0 1")
    move = chess.Move.from_uci("a7a8")
    assert index_to_move(move_to_index(move, board), board) == move


def test_mirror_move() -> None:
    assert mirror_move(chess.Move.from_uci("e2e4")) == chess.Move.from_uci("e7e5")
    assert mirror_move(chess.Move.from_uci("b7a8n")) == chess.Move.from_uci("b2a1n")
    drop = chess.Move(chess.E4, chess.E4, drop=chess.KNIGHT)
    assert mirror_move(drop) == chess.Move(chess.E5, chess.E5, drop=chess.KNIGHT)


@pytest.mark.parametrize(
    ("move", "message"),
    [
        (chess.Move.null(), "not representable"),
        (chess.Move(chess.E4, chess.E4, drop=chess.KNIGHT), "drops"),
        (chess.Move.from_uci("a1d2"), "not representable"),
        (chess.Move.from_uci("e2e4q"), "promotion is not a step"),
        (chess.Move.from_uci("e2e4n"), "promotion is not a step"),
        (chess.Move(chess.B7, chess.B8, promotion=chess.KING), "promotion piece"),
        (chess.Move(chess.B7, chess.B8, promotion=chess.PAWN), "promotion piece"),
    ],
)
def test_unrepresentable_moves_raise(move: chess.Move, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        move_to_index(move, chess.Board(positions.PROMOTION_WHITE))


@pytest.mark.parametrize(
    ("index", "message"),
    [
        (-1, "out of range"),
        (POLICY_SIZE, "out of range"),
        (0 * 64 + chess.H8, "on-board"),  # N, distance 1 from the last rank
        (64 * 64 + chess.B2, "on-board"),  # underpromotion plane, not from the 7th rank
    ],
)
def test_invalid_indices_raise(index: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        index_to_move(index, chess.Board())
