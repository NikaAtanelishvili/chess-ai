"""The puzzle benchmark: scoring, rating buckets, reading, sampling and the 500-puzzle fixture."""

import io
from collections import Counter
from pathlib import Path

import chess
import pytest
import zstandard

from chessnet.eval.players import PlayerSpec
from chessnet.eval.puzzles import (
    BUCKETS,
    Puzzle,
    bucket_of,
    read_puzzles,
    sample_puzzles,
    solve_puzzles,
    write_puzzles,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "puzzles" / "lichess_puzzles_500.csv"

# 1. Fool's mate: after the opponent's g2g4??, Qd8h4 mates.
FOOLS_MATE = Puzzle(
    "fool", "rnbqkbnr/pppp1ppp/8/4p3/8/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 2", ("g2g4", "d8h4"), 600
)
# 2. A back rank with two mates in one after the opponent's a7a6: Re8# (the solution) and Qd8#.
TWO_MATES = Puzzle("two", "6k1/p4ppp/8/8/8/8/5PPP/3QR1K1 b - - 0 1", ("a7a6", "e1e8"), 900)
# 3. The opponent puts its queen on the rook's file (b6d6??); the solution takes it.
HANGING = Puzzle("hang", "6k1/5ppp/1q6/8/8/8/5PPP/3R2K1 b - - 0 1", ("b6d6", "d1d6"), 1600)


def move(uci: str) -> chess.Move:
    return chess.Move.from_uci(uci)


def test_the_solution_is_accepted_and_other_moves_are_not() -> None:
    assert FOOLS_MATE.solved_by(move("d8h4"))
    assert not FOOLS_MATE.solved_by(move("g8f6"))
    assert not FOOLS_MATE.solved_by(move("d8h3"))  # not even legal
    assert HANGING.solved_by(move("d1d6"))
    assert not HANGING.solved_by(move("g1f1"))


def test_any_checkmate_counts_when_the_solution_mates() -> None:
    assert TWO_MATES.solved_by(move("e1e8"))
    assert TWO_MATES.solved_by(move("d1d8"))  # another mate in one
    assert not TWO_MATES.solved_by(move("e1e7"))


def test_the_player_sees_the_position_after_the_opponents_move() -> None:
    board = FOOLS_MATE.position()
    assert board.turn == chess.BLACK
    assert board.move_stack == [move("g2g4")]


def test_rating_buckets() -> None:
    assert [bucket_of(rating) for rating in (400, 1000, 1001, 1500, 1501, 2000, 2001, 3000)] == [
        "<=1000", "<=1000", "1001-1500", "1001-1500", "1501-2000", "1501-2000", ">2000", ">2000",
    ]  # fmt: skip


def synthetic_rows(count: int) -> list[Puzzle]:
    return [
        Puzzle(f"p{index}", FOOLS_MATE.fen, FOOLS_MATE.moves, 500 + 10 * index, 75, 90, 1000)
        for index in range(count)
    ] + [Puzzle("unstable", FOOLS_MATE.fen, FOOLS_MATE.moves, 1200, 300, 90, 1000)]


def test_read_write_and_compressed_files(tmp_path: Path) -> None:
    puzzles = synthetic_rows(5)
    plain = write_puzzles(puzzles, tmp_path / "puzzles.csv")
    assert list(read_puzzles(plain)) == puzzles
    compressed = tmp_path / "puzzles.csv.zst"
    compressed.write_bytes(zstandard.ZstdCompressor().compress(plain.read_bytes()))
    assert list(read_puzzles(compressed)) == puzzles


def test_sampling_is_stratified_filtered_and_reproducible(tmp_path: Path) -> None:
    path = write_puzzles(synthetic_rows(250), tmp_path / "puzzles.csv")  # ratings 500-2990
    first = sample_puzzles(path, 10, seed=3)
    assert first == sample_puzzles(path, 10, seed=3)
    assert first != sample_puzzles(path, 10, seed=4)
    assert Counter(bucket_of(p.rating) for p in first) == {label: 10 for label, _, _ in BUCKETS}
    assert all(puzzle.id != "unstable" for puzzle in first)  # rating deviation 300


def test_the_fixture_is_a_balanced_legal_sample() -> None:
    puzzles = list(read_puzzles(FIXTURE))
    assert len(puzzles) == 500
    assert Counter(bucket_of(p.rating) for p in puzzles) == {label: 125 for label, _, _ in BUCKETS}
    assert len({p.id for p in puzzles}) == 500
    for puzzle in puzzles:
        assert move(puzzle.moves[1]) in puzzle.position().legal_moves
        assert puzzle.rating_deviation <= 100


def test_solve_puzzles_counts_by_bucket() -> None:
    puzzles = [FOOLS_MATE, TWO_MATES, HANGING]
    lines: list[str] = []
    result = solve_puzzles(PlayerSpec("random", "random", seed=0), puzzles, report=lines.append)
    summary = result.summary()
    assert summary["puzzles"] == 3
    assert sum(bucket["puzzles"] for bucket in summary["buckets"].values()) == 3
    assert summary["buckets"]["<=1000"]["puzzles"] == 2
    assert 0 <= summary["accuracy"] <= 1
    assert lines[-1].startswith("[puzzles, random] 3/3")


def test_a_puzzle_needs_two_moves() -> None:
    with pytest.raises(IndexError):
        Puzzle("short", FOOLS_MATE.fen, ("g2g4",), 600).solved_by(move("d8h4"))


def test_csv_text_round_trip() -> None:
    buffer = io.StringIO(",".join(FOOLS_MATE.row()))
    assert buffer.getvalue().startswith("fool,")
