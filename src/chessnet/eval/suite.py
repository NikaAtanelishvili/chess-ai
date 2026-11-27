"""Evaluation suites: which matches and puzzle runs, how long they take, where results go.

``quick`` (about 10 minutes on a laptop CPU): policy against random, MCTS 400 against the
policy, and the 500-puzzle sample with the policy and MCTS 200. ``full``: more games of
both, MCTS 800 against Stockfish at its weakest settings (when Stockfish is installed),
optionally a second network against the first, and puzzles with the policy and MCTS 800.
Results go to ``<out>/<checkpoint>.<suite>.json`` (rewritten after every step, with the
parts not run yet), PGNs to ``<out>/<checkpoint>.<suite>/``, and a summary table to the
model card.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chessnet import __version__
from chessnet.eval.arena import run_match, write_pgn
from chessnet.eval.elo import format_elo
from chessnet.eval.openings import load_openings
from chessnet.eval.players import PlayerSpec, uci_option_range
from chessnet.eval.puzzles import BUCKETS, Puzzle, read_puzzles, sample_puzzles, solve_puzzles
from chessnet.eval.workers import Watchdog
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
    policy_safe = PlayerSpec("policy-safe", "policy-safe", weights=weights)
    mcts400 = PlayerSpec("mcts-400", "mcts", weights=weights, simulations=400)
    mcts800 = PlayerSpec("mcts-800", "mcts", weights=weights, simulations=800)
    full = suite == "full"
    matches = [
        MatchPlan("policy vs random", "policy-vs-random", policy, random_player,
                  100 if full else 20, "score >= 98%"),
        MatchPlan("policy-safe vs random", "policy-safe-vs-random", policy_safe, random_player,
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


def puzzle_slug(player: PlayerSpec) -> str:
    """The name of a puzzle run for ``--only``, e.g. 'puzzles-mcts-800'."""
    return f"puzzles-{player.name}"


def run_suite(
    plan: SuitePlan,
    weights: str,
    *,
    workers: int = 1,
    device: str = "cpu",
    out: Path = Path("eval/results"),
    model_card: Path | None = None,
    report: Callable[[str], None] = print,
    only: set[str] | None = None,
    watchdog: Watchdog | None = None,
    max_wall_time: float | None = None,
) -> dict[str, Any]:
    """Run every match and puzzle run of ``plan``; returns (and saves) the results.

    The results file is rewritten after every match and puzzle run, and lists the parts of
    the plan that have not run (``not_run``). ``only`` (match slugs and puzzle-run slugs
    such as 'puzzles-policy') runs just those parts and merges them into the suite's
    existing results file; each part records where and when it ran. With a ``watchdog``
    games and puzzles run in supervised workers. ``max_wall_time`` (seconds) stops the suite
    cleanly: the running part is saved as incomplete, and the rest is left as not run.
    """
    stem = Path(weights).stem
    results_path = out / f"{stem}.{plan.name}.json"
    matches, players = plan.matches, plan.puzzle_players
    if only is not None:
        known = {match.slug for match in matches} | {puzzle_slug(p) for p in players}
        unknown = only - known
        if unknown:
            raise ValueError(
                f"no match or puzzle run {', '.join(sorted(unknown))} in the {plan.name} suite "
                f"(choose from {', '.join(sorted(known))})"
            )
        matches = [match for match in matches if match.slug in only]
        players = [player for player in players if puzzle_slug(player) in only]
    if max_wall_time is not None:
        watchdog = replace(watchdog or Watchdog(), deadline=time.monotonic() + max_wall_time)
    merging = only is not None and results_path.exists()
    if merging:
        results = json.loads(results_path.read_text(encoding="utf-8"))
        if results["sha256"] != sha256(Path(weights)):
            raise ValueError(f"{results_path} holds results for another checkpoint")
    else:
        results = {
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
            "notes": list(plan.notes),
            "matches": [],
            "puzzles": [],
        }
    out.mkdir(parents=True, exist_ok=True)
    run = _Run(plan, results, results_path, out / f"{stem}.{plan.name}", workers, device,
               watchdog, report, keep_wall_time=merging)  # fmt: skip
    run.save()
    if run.matches(matches):
        run.puzzles(players)
    if model_card is not None:
        update_model_card(model_card, results)
    _log_scalars(out / "tb" / f"{stem}.{plan.name}", results)
    report(f"results: {results_path}")
    return results


def _same_match(entry: dict[str, Any], match: MatchPlan) -> bool:
    """Entries from before slugs were recorded are matched by name."""
    return entry.get("slug", entry["name"]) in (match.slug, match.name)


@dataclass
class _Run:
    """Plays matches and puzzle runs into a results document, saving after each one."""

    plan: SuitePlan
    results: dict[str, Any]
    path: Path
    games_dir: Path
    workers: int
    device: str
    watchdog: Watchdog | None
    report: Callable[[str], None]
    keep_wall_time: bool = False
    started: float = field(default_factory=time.monotonic)

    def out_of_time(self) -> bool:
        deadline = self.watchdog.deadline if self.watchdog else None
        return deadline is not None and time.monotonic() >= deadline

    def provenance(self) -> dict[str, Any]:
        return {
            "device": resolved_device(self.device),
            "hardware": hardware(self.device),
            "workers": self.workers,
            "date": datetime.now(UTC).date().isoformat(),
            "git_commit": git_commit(),
        }

    def save(self) -> None:
        results = self.results
        missing = [m.name for m in self.plan.matches
                   if not any(_same_match(entry, m) for entry in results["matches"])]  # fmt: skip
        done = {run["player"]["name"] for run in results["puzzles"]}
        missing += [f"puzzles ({p.name})" for p in self.plan.puzzle_players if p.name not in done]
        results["not_run"] = missing
        if not self.keep_wall_time:
            results["wall_seconds"] = round(time.monotonic() - self.started, 1)
        self.path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    def matches(self, matches: list[MatchPlan]) -> bool:
        """Play ``matches`` (kept in the plan's order); False if the wall time ran out."""
        openings = load_openings()
        order = self.plan.matches
        for match in matches:
            if self.out_of_time():
                return False
            result = run_match(
                match.name, match.player, match.opponent, games=match.games, openings=openings,
                workers=self.workers, device=self.device, target=match.target,
                watchdog=self.watchdog, report=self.report,
            )  # fmt: skip
            pgn = write_pgn(result, self.games_dir / f"{match.slug}.pgn")
            entry = {"slug": match.slug, **result.summary(), "pgn": pgn.as_posix(),
                     **self.provenance()}  # fmt: skip
            kept = [m for m in self.results["matches"] if not _same_match(m, match)]
            rank = {plan.slug: index for index, plan in enumerate(order)}
            self.results["matches"] = sorted(
                [*kept, entry], key=lambda m: rank.get(m.get("slug", ""), _rank_by_name(m, order))
            )
            self.save()
            if result.incomplete:
                return False
        return True

    def puzzles(self, players: list[PlayerSpec]) -> bool:
        """Puzzle runs for ``players`` (in the plan's order); False if the wall time ran out."""
        names = [player.name for player in self.plan.puzzle_players]
        for player in players:
            if self.out_of_time():
                return False
            solved = solve_puzzles(
                player, self.plan.puzzles, workers=self.workers, device=self.device,
                source=self.plan.puzzle_source, watchdog=self.watchdog, report=self.report,
            )  # fmt: skip
            entry = {**solved.summary(), **self.provenance()}
            kept = [r for r in self.results["puzzles"] if r["player"]["name"] != player.name]
            self.results["puzzles"] = sorted(
                [*kept, entry],
                key=lambda r: (
                    names.index(r["player"]["name"]) if r["player"]["name"] in names else len(names)
                ),
            )
            self.save()
            if solved.incomplete:
                return False
        return True


def _rank_by_name(entry: dict[str, Any], order: list[MatchPlan]) -> int:
    names = [match.name for match in order]
    return names.index(entry["name"]) if entry["name"] in names else len(names)


def _log_scalars(directory: Path, results: dict[str, Any]) -> None:
    """Scores and puzzle accuracies as TensorBoard scalars, if TensorBoard is installed."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return
    with SummaryWriter(log_dir=str(directory)) as writer:
        for match in results["matches"]:
            if match["score"] is not None:  # None: no game finished
                writer.add_scalar(f"eval/score/{match['name']}", match["score"])
        for run in results["puzzles"]:
            if run["accuracy"] is not None:
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
        name = match["name"] + (" (incomplete)" if match.get("incomplete") else "")
        if match["score"] is None:
            score, elo = "-", "-"
        else:
            low, high = match["elo_95"]
            interval = f"{_elo_text(low, False)} to {_elo_text(high, True)}"
            score = f"{match['score']:.1%}"
            elo = f"{_elo_text(match['elo'], match['score'] >= 0.5)} ({interval})"
        lines.append(
            f"| {name} | {match['games']} | {match['wins']} / {match['draws']} / "
            f"{match['losses']} | {score} | {elo} | {match['target'] or '-'} |"
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
            name = run["player"]["name"] + (" (incomplete)" if run.get("incomplete") else "")
            cells = [_percent(run["buckets"][label]["accuracy"]) for label in labels]
            lines.append(f"| {name} | " + " | ".join(cells) + f" | {_percent(run['accuracy'])} |")
    elsewhere = [
        f"{match['name']} was played on {match['hardware']} ({match['device']})."
        for match in results["matches"]
        if match.get("hardware", results["hardware"]) != results["hardware"]
    ]
    unfinished = [
        f"{match['errors']} game(s) of {match['name']} did not finish "
        f"({_first_error(match['failures'])}) and are not counted."
        for match in results["matches"]
        if match.get("errors")
    ] + [
        f"{run['errors']} puzzle(s) for {run['player']['name']} did not finish "
        f"({next(iter(run['failures'].values()))}) and are not counted."
        for run in results["puzzles"]
        if run.get("errors")
    ]
    stopped = [
        f"{match['name']} was stopped at the wall-time limit after {match['games']} of "
        f"{match['games'] + len(match['failures'])} games."
        for match in results["matches"]
        if match.get("incomplete")
    ] + [
        f"Puzzles ({run['player']['name']}) were stopped at the wall-time limit after "
        f"{run['puzzles'] + run['errors']} of {run['puzzles'] + run['errors'] + run['not_run']}."
        for run in results["puzzles"]
        if run.get("incomplete")
    ]
    # Semicolons: match names can contain commas.
    not_run = [f"Not run: {'; '.join(results['not_run'])}."] if results.get("not_run") else []
    for note in [*results["notes"], *elsewhere, *unfinished, *stopped, *not_run]:
        lines += ["", note]
    return "\n".join(lines) + "\n"


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def _first_error(failures: list[dict[str, Any]]) -> str:
    """The first reason a game did not finish other than the wall-time limit."""
    return next(failure["reason"] for failure in failures if not failure["stopped"])


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
