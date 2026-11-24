"""Property tests: perspective symmetry and move-index bijection on generated positions.

The colour-symmetry tests are the permanent guard against the original defect (Black-to-move
positions encoded without mirroring the board or the move index).
"""

from collections import Counter

import chess
import numpy as np
from hypothesis import given

from chessnet.encoding.board import encode_board
from chessnet.encoding.moves import index_to_move, legal_mask, mirror_move, move_to_index
from chessnet.encoding.packed import batch_from_records, pack_position, unpack_batch_torch
from chessnet.encoding.value import DRAW

from helpers.strategies import boards, boards_with_move, sample_positions

SAMPLE_SIZE = 10_000


def black_to_move(board: chess.Board) -> chess.Board:
    return board if board.turn == chess.BLACK else board.mirror()


@given(boards())
def test_black_to_move_encodes_like_its_mirror_image(board: chess.Board) -> None:
    black = black_to_move(board)
    white = black.mirror()
    assert white.turn == chess.WHITE
    assert encode_board(black).tobytes() == encode_board(white).tobytes()


@given(boards())
def test_move_indices_are_colour_symmetric(board: chess.Board) -> None:
    mirrored = board.mirror()
    for move in board.legal_moves:
        twin = mirror_move(move)
        assert twin in mirrored.legal_moves
        assert move_to_index(move, board) == move_to_index(twin, mirrored)


@given(boards())
def test_legal_moves_round_trip_with_unique_indices(board: chess.Board) -> None:
    moves = list(board.legal_moves)
    indices = [move_to_index(move, board) for move in moves]
    assert len(set(indices)) == len(indices)
    for move, index in zip(moves, indices, strict=True):
        assert index_to_move(index, board) == move


@given(boards())
def test_legal_mask_is_true_exactly_at_legal_moves(board: chess.Board) -> None:
    mask = legal_mask(board)
    assert mask.sum() == board.legal_moves.count()
    expected = sorted(move_to_index(move, board) for move in board.legal_moves)
    assert np.flatnonzero(mask).tolist() == expected


@given(boards_with_move())
def test_packed_records_are_colour_symmetric(example: tuple[chess.Board, chess.Move]) -> None:
    board, move = example
    record = pack_position(board, move, DRAW)[0]
    twin = pack_position(board.mirror(), mirror_move(move), DRAW)[0]
    for field in ("bb", "castle", "ep", "halfmove", "move", "wdl"):
        assert np.array_equal(record[field], twin[field]), field


@given(boards_with_move())
def test_unpacked_record_equals_encoded_board(example: tuple[chess.Board, chess.Move]) -> None:
    board, move = example
    planes = unpack_batch_torch(batch_from_records(pack_position(board, move, DRAW)))
    assert planes[0].numpy().tobytes() == encode_board(board).tobytes()


def test_all_legal_moves_in_a_large_sample_are_representable() -> None:
    seen: Counter[str] = Counter()
    checked = 0
    for board in sample_positions(SAMPLE_SIZE, seed=20261124):
        checked += 1
        indices = set()
        for move in board.legal_moves:
            indices.add(move_to_index(move, board))  # raises if not representable
            if board.is_castling(move):
                seen[f"castling {'white' if board.turn else 'black'}"] += 1
            if board.is_en_passant(move):
                seen[f"en passant {'white' if board.turn else 'black'}"] += 1
            if move.promotion == chess.QUEEN:
                seen["queen promotion"] += 1
            elif move.promotion is not None:
                seen["underpromotion"] += 1
        assert len(indices) == board.legal_moves.count()
    assert checked == SAMPLE_SIZE
    for kind in ("castling white", "castling black", "en passant white", "en passant black"):
        assert seen[kind] > 0, kind
    assert seen["queen promotion"] > 0
    assert seen["underpromotion"] > 0
