"""Generated positions: hypothesis strategies and a seeded sampler."""

import random
from collections.abc import Iterator

import chess
from hypothesis import strategies as st

from helpers.positions import PLAYOUT_STARTS


@st.composite
def boards(draw: st.DrawFn, max_plies: int = 120) -> chess.Board:
    """A position reached by up to ``max_plies`` random legal moves from a start position.

    Each move is drawn by hypothesis, so failures are reproducible and shrink to short games.
    """
    board = chess.Board(draw(st.sampled_from(PLAYOUT_STARTS)))
    for _ in range(draw(st.integers(min_value=0, max_value=max_plies))):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(moves[draw(st.integers(min_value=0, max_value=len(moves) - 1))])
    return board


@st.composite
def boards_with_move(draw: st.DrawFn) -> tuple[chess.Board, chess.Move]:
    """A generated position together with one of its legal moves."""
    board = draw(boards().filter(lambda b: any(b.legal_moves)))
    moves = list(board.legal_moves)
    return board, moves[draw(st.integers(min_value=0, max_value=len(moves) - 1))]


def sample_positions(count: int, seed: int, max_plies: int = 200) -> Iterator[chess.Board]:
    """``count`` positions from seeded random games, cycling through the start positions."""
    rng = random.Random(seed)
    produced = 0
    game = 0
    while True:
        board = chess.Board(PLAYOUT_STARTS[game % len(PLAYOUT_STARTS)])
        game += 1
        for _ in range(max_plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            yield board.copy(stack=False)
            produced += 1
            if produced == count:
                return
            board.push(rng.choice(moves))
