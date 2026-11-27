"""Lichess puzzles: first-move accuracy by rating bucket.

A puzzle row gives a FEN, the moves (the opponent's move first, then the solution) and a
rating. The player sees the position after the opponent's move and must find the
solution's first move. As on Lichess, any checkmate counts when the solution checkmates.
"""

from __future__ import annotations

import csv
import io
import random
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess

from chessnet.engine.player import Player
from chessnet.eval.players import PlayerSpec
from chessnet.eval.workers import Failed, Watchdog, map_with_players

#: Rating buckets (inclusive bounds).
BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("<=1000", 0, 1000),
    ("1001-1500", 1001, 1500),
    ("1501-2000", 1501, 2000),
    (">2000", 2001, 100_000),
)
COLUMNS = (
    "PuzzleId",
    "FEN",
    "Moves",
    "Rating",
    "RatingDeviation",
    "Popularity",
    "NbPlays",
    "Themes",
    "GameUrl",
)


def bucket_of(rating: int) -> str:
    for label, low, high in BUCKETS:
        if low <= rating <= high:
            return label
    raise ValueError(f"rating {rating} is in no bucket")


@dataclass(frozen=True)
class Puzzle:
    id: str
    fen: str
    moves: tuple[str, ...]
    rating: int
    rating_deviation: int = 0
    popularity: int = 0
    plays: int = 0
    themes: str = ""
    url: str = ""

    def position(self) -> chess.Board:
        """The position the player sees: after the opponent's first move."""
        board = chess.Board(self.fen)
        board.push_uci(self.moves[0])
        return board

    def solved_by(self, move: chess.Move) -> bool:
        """The solution's first move, or any checkmate when the solution checkmates."""
        board = self.position()
        solution = chess.Move.from_uci(self.moves[1])
        if move == solution:
            return True
        if move not in board.legal_moves:
            return False
        after_solution, after_move = board.copy(stack=False), board.copy(stack=False)
        after_solution.push(solution)
        after_move.push(move)
        return after_solution.is_checkmate() and after_move.is_checkmate()

    def row(self) -> list[str]:
        return [
            self.id,
            self.fen,
            " ".join(self.moves),
            str(self.rating),
            str(self.rating_deviation),
            str(self.popularity),
            str(self.plays),
            self.themes,
            self.url,
        ]


def read_puzzles(path: Path) -> Iterator[Puzzle]:
    """Puzzles from a Lichess CSV, plain or zstd-compressed (``.zst``)."""
    with path.open("rb") as raw:
        if path.suffix == ".zst":
            import zstandard

            stream: Any = zstandard.ZstdDecompressor().stream_reader(raw)
        else:
            stream = raw
        for row in csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8", newline="")):
            yield Puzzle(
                id=row["PuzzleId"],
                fen=row["FEN"],
                moves=tuple(row["Moves"].split()),
                rating=int(row["Rating"]),
                rating_deviation=int(row.get("RatingDeviation") or 0),
                popularity=int(row.get("Popularity") or 0),
                plays=int(row.get("NbPlays") or 0),
                themes=row.get("Themes", ""),
                url=row.get("GameUrl", ""),
            )


def sample_puzzles(
    path: Path,
    per_bucket: int,
    *,
    seed: int = 0,
    max_deviation: int = 100,
    min_popularity: int = 80,
    min_plays: int = 200,
) -> list[Puzzle]:
    """``per_bucket`` puzzles from each rating bucket, uniformly among the well-established
    ones (stable rating, popular, often played), reproducible for a given file and seed."""
    rng = random.Random(seed)
    reservoirs: dict[str, list[Puzzle]] = {label: [] for label, _, _ in BUCKETS}
    seen = dict.fromkeys(reservoirs, 0)
    for puzzle in read_puzzles(path):
        if (
            puzzle.rating_deviation > max_deviation
            or puzzle.popularity < min_popularity
            or puzzle.plays < min_plays
            or len(puzzle.moves) < 2
        ):
            continue
        label = bucket_of(puzzle.rating)
        seen[label] += 1
        reservoir = reservoirs[label]
        if len(reservoir) < per_bucket:
            reservoir.append(puzzle)
        else:
            slot = rng.randrange(seen[label])
            if slot < per_bucket:
                reservoir[slot] = puzzle
    chosen = [puzzle for label, _, _ in BUCKETS for puzzle in reservoirs[label]]
    return sorted(chosen, key=lambda puzzle: (puzzle.rating, puzzle.id))


def write_puzzles(puzzles: Sequence[Puzzle], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows(puzzle.row() for puzzle in puzzles)
    return path


def solve_task(
    players: tuple[Player, ...], puzzle: Puzzle, beat: Callable[[], None]
) -> tuple[str, bool, str]:
    """(puzzle id, solved, the move played) for one puzzle."""
    [player] = players
    move = player.choose(puzzle.position())
    beat()
    return puzzle.id, puzzle.solved_by(move), move.uci()


@dataclass
class PuzzleResult:
    player: PlayerSpec
    puzzles: list[Puzzle]
    solved: dict[str, bool] = field(default_factory=dict)
    seconds: float = 0.0
    source: str = ""
    #: Puzzles that failed (an error or a timeout): id -> reason.
    failures: dict[str, str] = field(default_factory=dict)
    #: Puzzles the wall-time limit stopped or never started.
    stopped: list[str] = field(default_factory=list)

    @property
    def incomplete(self) -> bool:
        return bool(self.stopped)

    def summary(self) -> dict[str, Any]:
        """Accuracy over the puzzles that finished; failures are listed, not counted."""
        buckets: dict[str, dict[str, float | None]] = {}
        done = [p for p in self.puzzles if p.id in self.solved]
        for label, _, _ in BUCKETS:
            ids = [p.id for p in done if bucket_of(p.rating) == label]
            right = sum(self.solved.get(puzzle_id, False) for puzzle_id in ids)
            buckets[label] = {
                "puzzles": len(ids),
                "solved": right,
                "accuracy": round(right / len(ids), 4) if ids else None,
            }
        total = sum(self.solved.values())
        return {
            "player": self.player.describe(),
            "source": self.source,
            "puzzles": len(done),
            "solved": total,
            "accuracy": round(total / len(done), 4) if done else None,
            "buckets": buckets,
            "incomplete": self.incomplete,
            "not_run": len(self.stopped),
            "errors": len(self.failures),
            "failures": self.failures,
            "seconds_per_puzzle": round(self.seconds / max(len(done), 1), 3),
            "wall_seconds": round(self.seconds, 1),
        }


def solve_puzzles(
    player: PlayerSpec,
    puzzles: Sequence[Puzzle],
    *,
    workers: int = 1,
    device: str = "cpu",
    source: str = "",
    watchdog: Watchdog | None = None,
    report: Callable[[str], None] = print,
) -> PuzzleResult:
    """First-move accuracy of ``player`` on ``puzzles`` (supervised with a ``watchdog``)."""
    result = PuzzleResult(player, list(puzzles), source=source)
    started = time.monotonic()
    label = f"puzzles, {player.name}"
    outcomes = map_with_players(
        (player,), solve_task, list(puzzles), workers=workers, device=device,
        watchdog=watchdog, label=label, report=report,
    )  # fmt: skip
    for done, outcome in enumerate(outcomes, start=1):
        if isinstance(outcome, Failed):
            if outcome.stopped:
                result.stopped.append(outcome.item.id)
            else:
                result.failures[outcome.item.id] = outcome.reason
            continue
        puzzle_id, solved, _ = outcome
        result.solved[puzzle_id] = solved
        if done % 50 == 0 or done == len(puzzles):
            right = sum(result.solved.values())
            report(f"[puzzles, {player.name}] {done}/{len(puzzles)}: {right} solved")
    result.seconds = time.monotonic() - started
    return result
