"""``chessnet preprocess``: stream a PGN dump into filtered, shuffled training shards.

The main process decompresses the stream, splits it into games and applies the header filter.
Games that pass go, in chunks, to parser processes that replay them with python-chess and
pack one record per position. Results are consumed in submission order, so a run is
deterministic for a given source, config and seed.
"""

from __future__ import annotations

import io
import logging
import multiprocessing
import os
import time
from collections import Counter, deque
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chess
import chess.pgn

from chessnet import __version__
from chessnet.config import DataConfig
from chessnet.data.filters import HeaderFilter, Rejection
from chessnet.data.pgn_stream import PGNSource
from chessnet.data.shards import (
    MANIFEST_NAME,
    RECORD_DTYPE_DESCRIPTION,
    GameCounts,
    Manifest,
    ShardWriter,
    SourceInfo,
    Split,
    SplitInfo,
    split_of_site,
    write_manifest,
)
from chessnet.encoding import ENCODING_VERSION
from chessnet.encoding.packed import (
    MAX_ELO,
    RECORD_SIZE,
    PackedRecords,
    empty_records,
    pack_into,
)
from chessnet.encoding.value import wdl_from_result
from chessnet.provenance import git_commit

log = logging.getLogger(__name__)


class PreprocessError(RuntimeError):
    """The run produced data that must not be used (see the message)."""


@dataclass(frozen=True, slots=True)
class GameOutcome:
    """Result of parsing one game: either a rejection or its packed positions."""

    rejection: Rejection | None
    split: Split | None = None
    records: PackedRecords | None = None
    skipped_positions: int = 0


class _MainlineReader(chess.pgn.BaseVisitor[tuple[dict[str, str], list[chess.Move]]]):
    """Collects the headers and mainline moves of a game; raises on any PGN error."""

    def begin_game(self) -> None:
        self.headers: dict[str, str] = {}
        self.moves: list[chess.Move] = []

    def visit_header(self, tagname: str, tagvalue: str) -> None:
        self.headers[tagname] = tagvalue

    def begin_variation(self) -> chess.pgn.SkipType:
        return chess.pgn.SKIP

    def visit_move(self, board: chess.Board, move: chess.Move) -> None:
        self.moves.append(move)

    def handle_error(self, error: Exception) -> None:
        raise error

    def result(self) -> tuple[dict[str, str], list[chess.Move]]:
        return self.headers, self.moves


def encode_game(block: bytes, min_plies: int) -> GameOutcome:
    """Parse one game and pack every position together with the move played from it."""
    text = block.decode("utf-8", errors="replace")
    try:
        parsed = chess.pgn.read_game(io.StringIO(text), Visitor=_MainlineReader)
    except ValueError:  # python-chess's illegal/ambiguous/invalid move errors
        return GameOutcome(Rejection.PARSE_ERROR)
    if parsed is None:
        return GameOutcome(Rejection.MALFORMED)
    headers, moves = parsed
    if len(moves) < min_plies:
        return GameOutcome(Rejection.TOO_SHORT)

    wdl = {color: wdl_from_result(headers["Result"], color) for color in chess.COLORS}
    elo = {
        chess.WHITE: min(int(headers["WhiteElo"]), MAX_ELO),
        chess.BLACK: min(int(headers["BlackElo"]), MAX_ELO),
    }
    records = empty_records(len(moves))
    board = chess.Board()
    written = 0
    for move in moves:
        try:
            pack_into(records, written, board, move, wdl[board.turn], elo=elo[board.turn])
            written += 1
        except ValueError:
            log.error("unrepresentable move %s in %s", move.uci(), headers.get("Site"))
        board.push(move)
    return GameOutcome(
        None,
        split_of_site(headers["Site"]),
        records[:written],
        skipped_positions=len(moves) - written,
    )


def encode_games(blocks: list[bytes], min_plies: int) -> list[GameOutcome]:
    """Worker task: :func:`encode_game` for a chunk of games."""
    return [encode_game(block, min_plies) for block in blocks]


class _InlineExecutor(Executor):
    """Runs tasks immediately in the calling process (``workers=0``: tests, debugging)."""

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        future: Future[Any] = Future()
        future.set_result(fn(*args, **kwargs))
        return future


@dataclass
class _Progress:
    """Counters for progress lines and the manifest."""

    source: PGNSource
    max_games: int | None
    every: float
    started: float
    scanned: int = 0
    passed: int = 0
    kept: int = 0
    skipped_positions: int = 0
    rejected: Counter[Rejection] = field(default_factory=Counter)
    games: Counter[Split] = field(default_factory=Counter)
    positions: Counter[Split] = field(default_factory=Counter)
    last_report: float = 0.0

    def maybe_report(self, now: float) -> None:
        if now - max(self.last_report, self.started) >= self.every:
            self.report(now)

    def report(self, now: float, *, final: bool = False) -> None:
        self.last_report = now
        elapsed = max(now - self.started, 1e-9)
        read_mb = self.source.bytes_read / 1e6
        line = (
            f"read {read_mb:,.0f} MB ({read_mb / elapsed:.1f} MB/s, "
            f"{self.source.bytes_decompressed / 1e6 / elapsed:.1f} MB/s decompressed) | "
            f"scanned {self.scanned:,} games ({self.scanned / elapsed:,.0f}/s) | "
            f"kept {self.kept:,} ({self.kept / elapsed:,.0f}/s, "
            f"{self.kept / max(self.scanned, 1):.1%}) | "
            f"{sum(self.positions.values()):,} positions"
        )
        if self.max_games and self.kept and not final:
            remaining = (self.max_games - self.kept) * elapsed / self.kept
            line += f" | ETA {remaining / 60:,.0f} min"
        log.info(line)


def _candidate_chunks(
    source: PGNSource, header_filter: HeaderFilter, progress: _Progress, chunk_games: int
) -> Iterator[list[bytes]]:
    """Games that pass the header filter, in chunks of ``chunk_games``."""
    chunk: list[bytes] = []
    for block in source.games():
        progress.scanned += 1
        rejection = header_filter.rejection(block)
        if rejection is not None:
            progress.rejected[rejection] += 1
            continue
        progress.passed += 1
        chunk.append(block)
        if len(chunk) == chunk_games:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def preprocess(
    config: DataConfig,
    source: str,
    out_dir: Path,
    *,
    max_games: int | None = None,
    workers: int | None = None,
    progress_every: float = 30.0,
    chunk_games: int = 64,
) -> Manifest:
    """Build a dataset directory (shards plus ``manifest.json``) from a PGN dump.

    ``max_games`` (default: the config's) stops the run once that many games are kept.
    ``workers`` is the number of parser processes (default: the CPU count; 0 parses in this
    process). The manifest is written last, so a directory without one is incomplete.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"output directory {out_dir} is not empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    if max_games is None:
        max_games = config.max_games
    if workers is None:
        workers = os.cpu_count() or 1
    started = time.monotonic()
    log.info("preprocess %s -> %s (max games %s, %d workers)", source, out_dir, max_games, workers)

    header_filter = HeaderFilter(config.filters)
    writer = ShardWriter(
        out_dir, num_buckets=config.num_buckets, shard_size=config.shard_size, seed=config.seed
    )
    stopped_because = "end_of_source"
    with PGNSource(source) as pgn:
        progress = _Progress(pgn, max_games, progress_every, started)

        def consume(outcomes: list[GameOutcome]) -> bool:
            """Record a chunk's outcomes; True once ``max_games`` games are kept."""
            for outcome in outcomes:
                if outcome.rejection is not None:
                    progress.rejected[outcome.rejection] += 1
                    continue
                assert outcome.split is not None
                assert outcome.records is not None
                writer.add(outcome.split, outcome.records)
                progress.kept += 1
                progress.games[outcome.split] += 1
                progress.positions[outcome.split] += len(outcome.records)
                progress.skipped_positions += outcome.skipped_positions
                if max_games is not None and progress.kept >= max_games:
                    return True
            return False

        executor: Executor = (
            ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
            if workers > 0
            else _InlineExecutor()
        )
        try:
            in_flight: deque[Future[list[GameOutcome]]] = deque()
            max_in_flight = 2 * max(workers, 1)
            done = False
            for chunk in _candidate_chunks(pgn, header_filter, progress, chunk_games):
                in_flight.append(executor.submit(encode_games, chunk, config.filters.min_plies))
                while len(in_flight) >= max_in_flight and not done:
                    done = consume(in_flight.popleft().result())
                progress.maybe_report(time.monotonic())
                if done:
                    break
            while in_flight and not done:
                done = consume(in_flight.popleft().result())
            if done:
                stopped_because = "max_games"
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        progress.report(time.monotonic(), final=True)
        source_info = SourceInfo(
            location=source,
            kind="url" if pgn.is_url else "path",
            compressed=pgn.compressed,
            bytes_read=pgn.bytes_read,
            bytes_decompressed=pgn.bytes_decompressed,
            stopped_because=stopped_because,
        )

    if progress.skipped_positions:
        raise PreprocessError(
            f"{progress.skipped_positions} positions had unrepresentable moves; the move "
            "encoding is incomplete, so this dataset must not be used"
        )

    log.info("shuffling %s positions into shards", f"{sum(progress.positions.values()):,}")
    shards = writer.finalize()
    manifest = Manifest(
        encoding_version=ENCODING_VERSION,
        record_size=RECORD_SIZE,
        record_dtype=RECORD_DTYPE_DESCRIPTION,
        chessnet_version=__version__,
        git_commit=git_commit(),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        elapsed_seconds=round(time.monotonic() - started, 1),
        config=config,
        source=source_info,
        games=GameCounts(
            scanned=progress.scanned,
            passed_header_filter=progress.passed,
            kept=progress.kept,
            rejected={str(reason): count for reason, count in sorted(progress.rejected.items())},
        ),
        skipped_positions=progress.skipped_positions,
        splits={
            split: SplitInfo(
                games=progress.games[split],
                positions=progress.positions[split],
                shards=tuple(shards[split]),
            )
            for split in Split
        },
    )
    write_manifest(out_dir, manifest)
    log.info("wrote %s", out_dir / MANIFEST_NAME)
    return manifest
