"""Matches between two players: alternating colours over an opening book, Elo, PGNs.

Each opening is played twice, once with each colour, so deterministic players still play
different games. A game ends on checkmate, stalemate, insufficient material, threefold
repetition (on the board, not merely available) or the 50-move rule, or as a draw after
``max_plies``. An illegal move loses at once.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chess
import chess.pgn

from chessnet.engine.player import Player
from chessnet.eval.elo import MatchStats
from chessnet.eval.openings import Opening
from chessnet.eval.players import PlayerSpec
from chessnet.eval.workers import Failed, Watchdog, map_with_players

MAX_PLIES = 600


@dataclass(frozen=True)
class GameTask:
    index: int
    opening: str
    fen: str
    player_white: bool
    max_plies: int = MAX_PLIES


@dataclass(frozen=True)
class GameRecord:
    """One finished game; ``result`` is from White's side, as in PGN."""

    index: int
    opening: str
    fen: str
    player_white: bool
    moves: tuple[str, ...]
    result: str
    termination: str
    player_seconds: float
    opponent_seconds: float

    @property
    def player_points(self) -> float:
        if self.result == "1/2-1/2":
            return 0.5
        white_won = self.result == "1-0"
        return 1.0 if white_won == self.player_white else 0.0

    @property
    def player_moves(self) -> int:
        """Moves the player made (it moves first when White, as every opening has White to move)."""
        return (len(self.moves) + (1 if self.player_white else 0)) // 2


def play_game(
    white: Player,
    black: Player,
    start: chess.Board,
    *,
    max_plies: int = MAX_PLIES,
    on_move: Callable[[], None] | None = None,
) -> tuple[chess.Board, str, str, dict[chess.Color, float]]:
    """Play from ``start``; returns the board, the result, the termination and thinking times.

    ``on_move`` is called after every move (the supervisor's sign of progress).
    """
    board = start.copy(stack=False)
    thinking = {chess.WHITE: 0.0, chess.BLACK: 0.0}
    while True:
        outcome = board.outcome()
        if outcome is not None:
            return board, outcome.result(), outcome.termination.name.lower(), thinking
        if board.is_repetition(3):
            return board, "1/2-1/2", "threefold_repetition", thinking
        if board.is_fifty_moves():
            return board, "1/2-1/2", "fifty_moves", thinking
        if len(board.move_stack) >= max_plies:
            return board, "1/2-1/2", "move_limit", thinking
        mover = board.turn
        started = time.perf_counter()
        move = (white if mover == chess.WHITE else black).choose(board)
        thinking[mover] += time.perf_counter() - started
        if move not in board.legal_moves:
            return board, "0-1" if mover == chess.WHITE else "1-0", "illegal_move", thinking
        board.push(move)
        if on_move is not None:
            on_move()


def play_task(players: tuple[Player, ...], task: GameTask, beat: Callable[[], None]) -> GameRecord:
    """Play one scheduled game: ``players`` is (player, opponent)."""
    player, opponent = players
    white, black = (player, opponent) if task.player_white else (opponent, player)
    board, result, termination, thinking = play_game(
        white, black, chess.Board(task.fen), max_plies=task.max_plies, on_move=beat
    )
    player_colour = chess.WHITE if task.player_white else chess.BLACK
    return GameRecord(
        index=task.index,
        opening=task.opening,
        fen=task.fen,
        player_white=task.player_white,
        moves=tuple(move.uci() for move in board.move_stack),
        result=result,
        termination=termination,
        player_seconds=thinking[player_colour],
        opponent_seconds=thinking[not player_colour],
    )


def schedule(games: int, openings: Sequence[Opening], max_plies: int = MAX_PLIES) -> list[GameTask]:
    """Game i uses opening i // 2 (cycling); the player has White in the even games."""
    if not openings:
        raise ValueError("no openings")
    return [
        GameTask(
            index=index,
            opening=openings[(index // 2) % len(openings)].name,
            fen=openings[(index // 2) % len(openings)].fen,
            player_white=index % 2 == 0,
            max_plies=max_plies,
        )
        for index in range(games)
    ]


@dataclass
class MatchResult:
    name: str
    player: PlayerSpec
    opponent: PlayerSpec
    records: list[GameRecord] = field(default_factory=list)
    seconds: float = 0.0
    target: str | None = None
    #: Games that did not finish: {"game": number, "opening", "reason", "stopped"}.
    failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def incomplete(self) -> bool:
        """The wall-time limit stopped the match before all its games were played."""
        return any(failure["stopped"] for failure in self.failures)

    @property
    def stats(self) -> MatchStats:
        points = [record.player_points for record in self.records]
        return MatchStats(points.count(1.0), points.count(0.5), points.count(0.0))

    def summary(self) -> dict[str, Any]:
        stats = self.stats
        score: float | None = None
        elo: float | None = None
        elo_95: list[float | None] = [None, None]
        if stats.games:  # no finished game, no score
            low, high = stats.elo_interval
            score, elo, elo_95 = (
                round(stats.score, 4),
                _finite(stats.elo),
                [_finite(low), _finite(high)],
            )
        moves = sum(record.player_moves for record in self.records) or 1
        opponent_moves = sum(len(r.moves) - r.player_moves for r in self.records) or 1
        return {
            "name": self.name,
            "player": self.player.describe(),
            "opponent": self.opponent.describe(),
            "games": stats.games,
            "wins": stats.wins,
            "draws": stats.draws,
            "losses": stats.losses,
            "score": score,
            "elo": elo,
            "elo_95": elo_95,
            "target": self.target,
            "incomplete": self.incomplete,
            "errors": sum(not failure["stopped"] for failure in self.failures),
            "failures": self.failures,
            "terminations": dict(Counter(record.termination for record in self.records)),
            "mean_plies": round(sum(len(r.moves) for r in self.records) / max(stats.games, 1), 1),
            "player_seconds_per_move": round(
                sum(r.player_seconds for r in self.records) / moves, 3
            ),
            "opponent_seconds_per_move": round(
                sum(r.opponent_seconds for r in self.records) / opponent_moves, 3
            ),
            "wall_seconds": round(self.seconds, 1),
        }


def _finite(value: float) -> float | None:
    """JSON has no infinity: an unbounded Elo becomes null."""
    return None if value in (float("inf"), float("-inf")) else round(value, 1)


def run_match(
    name: str,
    player: PlayerSpec,
    opponent: PlayerSpec,
    *,
    games: int,
    openings: Sequence[Opening],
    workers: int = 1,
    device: str = "cpu",
    max_plies: int = MAX_PLIES,
    target: str | None = None,
    watchdog: Watchdog | None = None,
    report: Callable[[str], None] = print,
) -> MatchResult:
    """Play ``games`` games of ``player`` against ``opponent``.

    With a ``watchdog`` the games run in supervised worker processes: a game that fails,
    hangs or is cut off by the deadline is recorded in ``failures`` and not counted.
    """
    result = MatchResult(name, player, opponent, target=target)
    tasks = schedule(games, openings, max_plies)
    started = time.monotonic()
    for record in map_with_players(
        (player, opponent), play_task, tasks, workers=workers, device=device,
        watchdog=watchdog, label=name, report=report,
    ):  # fmt: skip
        if isinstance(record, Failed):
            task = record.item
            result.failures.append(
                {"game": task.index + 1, "opening": task.opening, "reason": record.reason,
                 "stopped": record.stopped}
            )  # fmt: skip
            report(f"[{name}] game {task.index + 1}: not finished ({record.reason})")
            continue
        result.records.append(record)
        stats = result.stats
        report(
            f"[{name}] game {record.index + 1}: {record.result} ({record.termination}, "
            f"{len(record.moves)} plies); {player.name} {stats.wins + stats.draws / 2:g}/"
            f"{stats.games}"
        )
    result.records.sort(key=lambda record: record.index)
    result.failures.sort(key=lambda failure: failure["game"])
    result.seconds = time.monotonic() - started
    return result


def write_pgn(result: MatchResult, path: Path) -> Path:
    """All games of a match in one PGN file, in schedule order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    date = datetime.now(UTC).strftime("%Y.%m.%d")
    games = []
    for record in result.records:
        game = chess.pgn.Game()
        game.setup(chess.Board(record.fen))
        node: chess.pgn.GameNode = game
        for uci in record.moves:
            node = node.add_variation(chess.Move.from_uci(uci))
        white, black = (
            (result.player.name, result.opponent.name)
            if record.player_white
            else (result.opponent.name, result.player.name)
        )
        game.headers.update(
            {
                "Event": result.name,
                "Site": "chessnet arena",
                "Date": date,
                "Round": str(record.index + 1),
                "White": white,
                "Black": black,
                "Result": record.result,
                "Termination": record.termination,
                "Opening": record.opening,
            }
        )
        games.append(str(game))
    path.write_text("\n\n".join(games) + "\n", encoding="utf-8")  # one blank line between games
    return path
