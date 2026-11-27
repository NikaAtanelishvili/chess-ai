"""Evaluation suites: which matches and puzzle runs, how long they take, where results go.

``quick`` (about 10 minutes on a laptop CPU): policy against random, MCTS 400 against the
policy, and the 500-puzzle sample with the policy and MCTS 200. ``full``: more games of
both, MCTS 800 against Stockfish at its weakest settings (when Stockfish is installed),
optionally a second network against the first, and puzzles with the policy and MCTS 800.
Results go to ``<out>/<checkpoint>.<suite>.json`` (rewritten after every step), PGNs to
``<out>/<checkpoint>.<suite>/``, and a summary table to the model card.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chessnet import __version__
from chessnet.eval.arena import run_match, write_pgn
from chessnet.eval.elo import format_elo
from chessnet.eval.openings import load_openings
from chessnet.eval.players import PlayerSpec, uci_option_range
from chessnet.eval.puzzles import BUCKETS, Puzzle, read_puzzles, sample_puzzles, solve_puzzles
from chessnet.provenance import git_commit

REPOSITORY = Path(__file__).resolve().parents[3]
FIXTURE_PUZZLES = REPOSITORY / "tests" / "fixtures" / "puzzles" / "lichess_puzzles_500.csv"
#: Puzzles per rating bucket sampled from the full Lichess database in the full suite.
FULL_PUZZLES_PER_BUCKET = 500
#: Seconds per move for Stockfish, and for both sides in the equal-time comparison.
STOCKFISH_MOVETIME = 1.0
EQUAL_MOVETIME = 1.0


@dataclass(frozen=True)
class MatchPlan:
    name: str
    slug: str
    player: PlayerSpec
    opponent: PlayerSpec
    games: int
    target: str | None = None


@dataclass
class SuitePlan:
    name: str
    matches: list[MatchPlan]
    puzzle_players: list[PlayerSpec]
    puzzles: list[Puzzle]
    puzzle_source: str
    notes: list[str] = field(default_factory=list)


def plan_suite(
    suite: str,
    weights: str,
    *,
    compare: str | None = None,
    stockfish: str | None = None,
    puzzles: Path | None = None,
) -> SuitePlan:
    """The matches and puzzle runs of ``suite`` ('quick' or 'full') for a network."""
    if suite not in ("quick", "full"):
        raise ValueError(f"unknown suite {suite!r}: use 'quick' or 'full'")
    random_player = PlayerSpec("random", "random", seed=0)
    policy = PlayerSpec("policy", "policy", weights=weights)
    mcts400 = PlayerSpec("mcts-400", "mcts", weights=weights, simulations=400)
    mcts800 = PlayerSpec("mcts-800", "mcts", weights=weights, simulations=800)
    full = suite == "full"
    matches = [
        MatchPlan("policy vs random", "policy-vs-random", policy, random_player,
                  100 if full else 20, "score >= 98%"),
        MatchPlan("MCTS 400 vs policy", "mcts400-vs-policy", mcts400, policy,
                  100 if full else 20, "score > 55%"),
    ]  # fmt: skip
    notes: list[str] = []
    if full and stockfish:
        command = (stockfish,)
        lowest, _ = uci_option_range(command, "UCI_Elo")
        common: tuple[tuple[str, str | int | bool], ...] = (("Threads", 1), ("Hash", 16))
        limited = PlayerSpec(
            f"stockfish-elo-{lowest}",
            "uci",
            command=command,
            options=(("UCI_LimitStrength", True), ("UCI_Elo", lowest), *common),
            movetime=STOCKFISH_MOVETIME,
        )
        skill0 = PlayerSpec(
            "stockfish-skill-0",
            "uci",
            command=command,
            options=(("Skill Level", 0), *common),
            movetime=STOCKFISH_MOVETIME,
        )
        matches += [
            MatchPlan(f"MCTS 800 vs Stockfish UCI_Elo {lowest}", "mcts800-vs-stockfish-elo",
                      mcts800, limited, 50, "non-zero score"),
            MatchPlan("MCTS 800 vs Stockfish Skill Level 0", "mcts800-vs-stockfish-skill0",
                      mcts800, skill0, 50),
        ]  # fmt: skip
    elif full:
        notes.append("Stockfish was not found, so its matches were skipped.")
    if full and compare:
        other = Path(compare).stem
        matches += [
            MatchPlan(f"{other} vs base, 800 playouts each", "compare-equal-nodes",
                      PlayerSpec(f"{other}-mcts-800", "mcts", weights=compare, simulations=800),
                      mcts800, 50),
            MatchPlan(f"{other} vs base, {EQUAL_MOVETIME:g} s per move each", "compare-equal-time",
                      PlayerSpec(f"{other}-mcts-{EQUAL_MOVETIME:g}s", "mcts", weights=compare,
                                 movetime=EQUAL_MOVETIME),
                      PlayerSpec(f"mcts-{EQUAL_MOVETIME:g}s", "mcts", weights=weights,
                                 movetime=EQUAL_MOVETIME),
                      50),
        ]  # fmt: skip
    source = puzzles or FIXTURE_PUZZLES
    if full and source.suffix == ".zst":
        puzzle_set = sample_puzzles(source, FULL_PUZZLES_PER_BUCKET, seed=0)
        description = f"{len(puzzle_set)} sampled from {source.name} (seed 0)"
    else:
        puzzle_set = list(read_puzzles(source))
        description = f"{len(puzzle_set)} from {source.name}"
    searcher = mcts800 if full else PlayerSpec("mcts-200", "mcts", weights=weights, simulations=200)
    return SuitePlan(suite, matches, [policy, searcher], puzzle_set, description, notes)


def _quiet(line: str) -> None:
    pass


def estimate_suite(
    plan: SuitePlan,
    *,
    workers: int = 1,
    device: str = "cpu",
    report: Callable[[str], None] = print,
) -> list[tuple[str, float, float]]:
    """Time two games of every match and ten puzzles of every puzzle run, and extrapolate.

    Returns (item, seconds per game or puzzle, estimated seconds for the suite), assuming
    ``workers`` parallel processes scale linearly.
    """
    openings = load_openings()
    rows: list[tuple[str, float, float]] = []
    for match in plan.matches:
        result = run_match(
            match.name, match.player, match.opponent, games=2, openings=openings,
            device=device, report=_quiet,
        )  # fmt: skip
        per_game = result.seconds / 2
        rows.append((f"{match.name} ({match.games} games)", per_game, per_game * match.games))
    for player in plan.puzzle_players:
        sample = plan.puzzles[:: max(1, len(plan.puzzles) // 10)][:10]
        solved = solve_puzzles(player, sample, device=device, report=_quiet)
        per_puzzle = solved.seconds / len(sample)
        label = f"puzzles, {player.name} ({len(plan.puzzles)})"
        rows.append((label, per_puzzle, per_puzzle * len(plan.puzzles)))
    total = sum(seconds for _, _, seconds in rows)
    report(f"Estimated wall time of the {plan.name} suite ({workers} worker(s)):")
    for label, unit, seconds in rows:
        report(f"  {label:58s} {unit:7.1f} s each   {seconds / workers / 60:7.1f} min")
    report(f"  {'total':58s} {'':15s}   {total / workers / 3600:7.2f} h")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolved_device(device: str) -> str:
    """'cuda' when the suite's workers will use GPUs, else 'cpu'."""
    if device == "cpu":
        return "cpu"
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def hardware(device: str) -> str:
    """The GPUs' name when the suite runs on them, else the CPU's model."""
    if resolved_device(device) == "cuda":
        import torch

        return f"{torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def run_suite(
    plan: SuitePlan,
    weights: str,
    *,
    workers: int = 1,
    device: str = "cpu",
    out: Path = Path("eval/results"),
    model_card: Path | None = None,
    report: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run every match and puzzle run of ``plan``; returns (and saves) the results."""
    stem = Path(weights).stem
    results_path = out / f"{stem}.{plan.name}.json"
    games_dir = out / f"{stem}.{plan.name}"
    results: dict[str, Any] = {
        "checkpoint": Path(weights).name,
        "sha256": sha256(Path(weights)),
        "suite": plan.name,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "chessnet_version": __version__,
        "git_commit": git_commit(),
        "device": resolved_device(device),
        "hardware": hardware(device),
        "workers": workers,
        "openings": "50 balanced openings, each played with both colours",
        "notes": plan.notes,
        "matches": [],
        "puzzles": [],
    }
    out.mkdir(parents=True, exist_ok=True)
    openings = load_openings()
    started = time.monotonic()
    for match in plan.matches:
        result = run_match(
            match.name, match.player, match.opponent, games=match.games, openings=openings,
            workers=workers, device=device, target=match.target, report=report,
        )  # fmt: skip
        pgn = write_pgn(result, games_dir / f"{match.slug}.pgn")
        results["matches"].append({**result.summary(), "pgn": str(pgn)})
        _save(results, results_path, started)
    for player in plan.puzzle_players:
        solved = solve_puzzles(
            player, plan.puzzles, workers=workers, device=device,
            source=plan.puzzle_source, report=report,
        )  # fmt: skip
        results["puzzles"].append(solved.summary())
        _save(results, results_path, started)
    if model_card is not None:
        update_model_card(model_card, results)
    _log_scalars(out / "tb" / f"{stem}.{plan.name}", results)
    report(f"results: {results_path}")
    return results


def _save(results: dict[str, Any], path: Path, started: float) -> None:
    results["wall_seconds"] = round(time.monotonic() - started, 1)
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def _log_scalars(directory: Path, results: dict[str, Any]) -> None:
    """Scores and puzzle accuracies as TensorBoard scalars, if TensorBoard is installed."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return
    with SummaryWriter(log_dir=str(directory)) as writer:
        for match in results["matches"]:
            writer.add_scalar(f"eval/score/{match['name']}", match["score"])
        for run in results["puzzles"]:
            writer.add_scalar(f"eval/puzzles/{run['player']['name']}", run["accuracy"])


def summary_table(results: dict[str, Any]) -> str:
    """The Markdown section for one suite run."""
    lines = [
        f"### {results['checkpoint']}, {results['suite']} suite",
        "",
        f"{results['created'][:10]}, {results['hardware']} ({results['device']}, "
        f"{results['workers']} worker(s)), chessnet {results['chessnet_version']}. "
        f"{results['openings'].capitalize()}.",
        "",
        "| Match | Games | W / D / L | Score | Elo (95% CI) | Target |",
        "|-------|------:|-----------|------:|--------------|--------|",
    ]
    for match in results["matches"]:
        low, high = match["elo_95"]
        elo = _elo_text(match["elo"], match["score"] >= 0.5)
        interval = f"{_elo_text(low, False)} to {_elo_text(high, True)}"
        lines.append(
            f"| {match['name']} | {match['games']} | {match['wins']} / {match['draws']} / "
            f"{match['losses']} | {match['score']:.1%} | {elo} ({interval}) | "
            f"{match['target'] or '-'} |"
        )
    if results["puzzles"]:
        labels = [label for label, _, _ in BUCKETS]
        lines += [
            "",
            f"Puzzles, first move: {results['puzzles'][0]['source']}.",
            "",
            "| Player | " + " | ".join(labels) + " | All |",
            "|--------|" + "---:|" * (len(labels) + 1),
        ]
        for run in results["puzzles"]:
            cells = [f"{run['buckets'][label]['accuracy']:.1%}" for label in labels]
            lines.append(
                f"| {run['player']['name']} | " + " | ".join(cells) + f" | {run['accuracy']:.1%} |"
            )
    for note in results["notes"]:
        lines += ["", note]
    return "\n".join(lines) + "\n"


def _elo_text(value: float | None, positive: bool) -> str:
    if value is None:
        return "+inf" if positive else "-inf"
    return format_elo(value)


def update_model_card(path: Path, results: dict[str, Any]) -> Path:
    """Put the run's table into the model card, replacing an earlier run of the same suite."""
    key = f"{Path(results['checkpoint']).stem} {results['suite']}"
    start, end = f"<!-- eval: {key} -->", f"<!-- end eval: {key} -->"
    section = f"{start}\n{summary_table(results)}{end}\n"
    text = path.read_text(encoding="utf-8") if path.exists() else "# Model card\n"
    if start in text and end in text:
        before, rest = text.split(start, 1)
        after = rest.split(end, 1)[1].lstrip("\n")
        text = before + section + ("\n" + after if after else "")
    else:
        text = text.rstrip("\n") + "\n\n" + section
    path.write_text(text, encoding="utf-8")
    return path
