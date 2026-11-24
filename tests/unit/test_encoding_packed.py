import random

import chess
import numpy as np
import pytest
import torch

from chessnet.encoding.board import encode_board, perspective_board
from chessnet.encoding.moves import index_to_move, move_to_index
from chessnet.encoding.packed import (
    PACKED_DTYPE,
    RECORD_SIZE,
    PackedBatch,
    batch_from_records,
    empty_records,
    pack_into,
    pack_position,
    unpack_batch_torch,
    unpack_board,
)
from chessnet.encoding.value import DRAW, LOSS, WIN, wdl_from_result

from helpers import positions

FOOLS_MATE = "f3 e5 g4 Qh4#"  # Black wins

# Short legal lines that together contain every kind of special move for both colours.
SPECIAL_LINES = [
    (positions.START, "e4 e5 Nf3 Nf6 Bc4 Bc5 O-O O-O"),
    (positions.START, "d4 d5 Nc3 Nc6 Bf4 Bf5 Qd2 Qd7 O-O-O O-O-O"),
    (positions.START, "e4 Nf6 e5 d5 exd6"),
    (positions.START, "a3 e5 a4 e4 d4 exd3"),
    (positions.START, FOOLS_MATE),
    (positions.PROMOTION_WHITE, "b8=Q"),
    (positions.PROMOTION_WHITE, "bxa8=N"),
    (positions.PROMOTION_WHITE, "bxc8=B"),
    (positions.PROMOTION_WHITE, "b8=R"),
    (positions.PROMOTION_BLACK, "b1=N"),
    (positions.PROMOTION_BLACK, "bxa1=Q"),
    (positions.PROMOTION_BLACK, "bxc1=R"),
    (positions.PROMOTION_BLACK, "b1=B"),
]


def played_positions(fen: str, line: str) -> list[tuple[chess.Board, chess.Move]]:
    """Each position of a line together with the move played from it."""
    board = chess.Board(fen)
    examples = []
    for san in line.split():
        move = board.parse_san(san)
        examples.append((board.copy(), move))
        board.push(move)
    return examples


def boards_with_moves(count: int, seed: int) -> list[tuple[chess.Board, chess.Move]]:
    """Fixture positions plus positions from seeded random games, each with a legal move."""
    rng = random.Random(seed)
    examples = []
    for fen in positions.ALL_FIXTURES:
        board = chess.Board(fen)
        examples.append((board, rng.choice(list(board.legal_moves))))
    game = 0
    while len(examples) < count:
        board = chess.Board(positions.PLAYOUT_STARTS[game % len(positions.PLAYOUT_STARTS)])
        game += 1
        while len(examples) < count and not board.is_game_over():
            move = rng.choice(list(board.legal_moves))
            examples.append((board.copy(), move))
            board.push(move)
    return examples


def pack_all(examples: list[tuple[chess.Board, chess.Move]]) -> np.ndarray:
    records = empty_records(len(examples))
    for i, (board, move) in enumerate(examples):
        pack_into(records, i, board, move, DRAW)
    return records


def test_record_layout() -> None:
    assert RECORD_SIZE == PACKED_DTYPE.itemsize == 106
    assert PACKED_DTYPE.names == ("bb", "castle", "ep", "halfmove", "move", "wdl", "ply", "elo")


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.gpu),
    ],
)
def test_unpack_of_pack_equals_encode(device: str) -> None:
    examples = boards_with_moves(500, seed=1)
    for fen, line in SPECIAL_LINES:
        examples += played_positions(fen, line)
    batch = batch_from_records(pack_all(examples)).to(device)
    planes = unpack_batch_torch(batch)
    assert planes.shape == (len(examples), 19, 8, 8)
    assert planes.dtype == torch.float32
    assert planes.device.type == device
    unpacked = planes.cpu().numpy()
    for row, (board, _) in enumerate(examples):
        assert unpacked[row].tobytes() == encode_board(board).tobytes(), board.fen()


def test_fields_round_trip() -> None:
    board = chess.Board(positions.EN_PASSANT_BLACK)
    board.halfmove_clock = 300  # clipped to 255 on disk, same feature value (1.0)
    move = chess.Move.from_uci("e4f3")
    records = pack_position(board, move, WIN, elo=2345)
    record = records[0]
    assert record["move"] == move_to_index(move, board)
    assert record["wdl"] == WIN
    assert record["ply"] == board.ply() == 5  # from the original (Black-to-move) board
    assert record["elo"] == 2345
    assert record["halfmove"] == 255
    assert record["ep"] == chess.F6  # Black's f3 target, seen from Black
    assert record["castle"] == 0b1111

    batch = batch_from_records(records)
    assert len(batch) == 1
    assert batch.move.dtype == torch.int64
    assert batch.move.item() == record["move"]
    assert (batch.wdl.item(), batch.ply.item(), batch.elo.item()) == (WIN, 5, 2345)
    assert np.array_equal(unpack_batch_torch(batch)[0].numpy(), encode_board(board))


@pytest.mark.parametrize("count", [1, 2, 7])
def test_batch_is_an_independent_copy(count: int) -> None:
    records = pack_all(boards_with_moves(count, seed=3))
    batch = batch_from_records(records)
    expected = unpack_batch_torch(batch).clone()
    records["bb"][:] = 0
    records["move"][:] = 0
    assert torch.equal(unpack_batch_torch(batch), expected)
    assert batch.bitboards.stride() == (12, 1)


def test_batch_to_device_keeps_values() -> None:
    batch = batch_from_records(pack_all(boards_with_moves(20, seed=2)))
    moved = batch.to("cpu", non_blocking=True)
    assert isinstance(moved, PackedBatch)
    for name in PackedBatch.__dataclass_fields__:
        assert torch.equal(getattr(moved, name), getattr(batch, name))


def test_value_perspective_black_win() -> None:
    examples = played_positions(positions.START, FOOLS_MATE)
    for board, move in examples:
        wdl = wdl_from_result("0-1", board.turn)
        record = pack_position(board, move, wdl)[0]
        expected = WIN if board.turn == chess.BLACK else LOSS
        assert wdl == record["wdl"] == expected
    final = examples[-1][0]
    final.push(examples[-1][1])
    assert final.is_checkmate()


@pytest.mark.parametrize(("fen", "line"), SPECIAL_LINES)
def test_policy_target_is_move_played_from_encoded_position(fen: str, line: str) -> None:
    """Regression guard: the legacy dataset labelled positions with the move that led to them.

    Each record must hold the move played *from* the encoded position, and that move must
    be legal there. Labelling the position after the move is rejected outright.
    """
    for board, move in played_positions(fen, line):
        record = pack_position(board, move, DRAW)[0]
        decoded = index_to_move(int(record["move"]), board)
        assert decoded == move
        assert decoded in board.legal_moves
        after = board.copy()
        after.push(move)
        with pytest.raises(ValueError, match="not legal"):
            pack_position(after, move, DRAW)


def test_special_lines_cover_every_special_move_for_both_colours() -> None:
    kinds = set()
    for fen, line in SPECIAL_LINES:
        for board, move in played_positions(fen, line):
            colour = board.turn
            if board.is_kingside_castling(move):
                kinds.add(("O-O", colour))
            if board.is_queenside_castling(move):
                kinds.add(("O-O-O", colour))
            if board.is_en_passant(move):
                kinds.add(("en passant", colour))
            if move.promotion:
                kinds.add((move.promotion, colour))
    for colour in (chess.WHITE, chess.BLACK):
        assert {("O-O", colour), ("O-O-O", colour), ("en passant", colour)} <= kinds
        for piece in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
            assert (piece, colour) in kinds


@pytest.mark.parametrize(
    ("move", "wdl", "elo", "message"),
    [
        (chess.Move.from_uci("e2e5"), DRAW, 0, "not legal"),
        (chess.Move.from_uci("e2e4"), 3, 0, "wdl"),
        (chess.Move.from_uci("e2e4"), -1, 0, "wdl"),
        (chess.Move.from_uci("e2e4"), DRAW, -1, "elo"),
        (chess.Move.from_uci("e2e4"), DRAW, 65536, "elo"),
    ],
)
def test_invalid_examples_raise(move: chess.Move, wdl: int, elo: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        pack_position(chess.Board(), move, wdl, elo=elo)


@pytest.mark.parametrize("fen", positions.ALL_FIXTURES)
def test_unpack_board_restores_the_movers_view(fen: str) -> None:
    board = chess.Board(fen)
    record = pack_position(board, next(iter(board.legal_moves)), DRAW)[0]
    view = unpack_board(record)
    assert view.turn == chess.WHITE
    assert view.fen() == perspective_board(board).fen()


def test_unpack_board_on_played_games() -> None:
    for fen, line in SPECIAL_LINES:
        for board, move in played_positions(fen, line):
            record = pack_position(board, move, DRAW)[0]
            view = unpack_board(record)
            assert view.fen() == perspective_board(board).fen()
            assert index_to_move(int(record["move"]), view) in view.legal_moves
