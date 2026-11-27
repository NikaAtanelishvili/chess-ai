"""Where MCTS stands on Stockfish's scale: a UCI_Elo ladder, and strength against playouts.

The ladder plays MCTS (800 playouts) against Stockfish at rising ``UCI_Elo`` levels, 1 s per
Stockfish move, and stops climbing after the first level where MCTS scores below
``stop_below``. The rating is the maximum-likelihood fit of the Elo model to every ladder
game, with a 95% likelihood-ratio interval. It is on Stockfish's ``UCI_Elo`` scale, which
Stockfish 16 anchors to CCRL engine ratings (goldfish 1.13 = 2000, fitted to games at 60 s +
0.6 s): an engine rating, not a FIDE or Lichess one.

The node-scaling curve plays MCTS at several playout budgets against one reference: the
ladder level where MCTS 800 scored closest to 50%, so that scores stay where Elo is measured
best. A point's rating is the reference level plus the Elo difference of its score; the 800
point reuses the ladder's match at that level.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from chessnet import __version__
from chessnet.eval.arena import run_match, write_pgn
from chessnet.eval.elo import format_elo
from chessnet.eval.openings import load_openings
from chessnet.eval.players import PlayerSpec
from chessnet.eval.suite import (
    STOCKFISH_MOVETIME,
    hardware,
    resolved_device,
    sha256,
    update_model_card,
)
from chessnet.eval.workers import Watchdog
from chessnet.provenance import git_commit

LADDER_LEVELS = (1500, 1700, 1900, 2100, 2300)
LADDER_GAMES = 40
STOP_BELOW = 0.25
LADDER_SIMULATIONS = 800
SCALING_SIMULATIONS = (50, 100, 200, 400, 800, 1600)
SCALING_GAMES = 32
SCALE = (
    "Stockfish 16's UCI_Elo scale, which it anchors to CCRL engine ratings (goldfish 1.13 = "
    "2000, games at 60 s + 0.6 s): an engine rating, not a FIDE or Lichess one"
)
#: A 95% likelihood-ratio interval: log-likelihood within chi-squared(1 df, 0.95) / 2 of the top.
_HALF_CHI2_95 = 3.841459 / 2
_LN10_400 = math.log(10) / 400


def stockfish_skill_level(elo: int) -> float:
    """The skill level Stockfish 16 plays at for a ``UCI_Elo`` (its own conversion)."""
    e = (elo - 1320) / (3190 - 1320)
    return min(max(((37.2473 * e - 40.8525) * e + 22.2943) * e - 0.311438, 0.0), 19.0)


def stockfish_at(
    command: Sequence[str], elo: int, movetime: float = STOCKFISH_MOVETIME
) -> PlayerSpec:
    """Stockfish limited to ``UCI_Elo`` (one thread, 16 MB hash, ``movetime`` per move)."""
    return PlayerSpec(
        f"stockfish-elo-{elo}",
        "uci",
        command=tuple(command),
        options=(("UCI_LimitStrength", True), ("UCI_Elo", elo), ("Threads", 1), ("Hash", 16)),
        movetime=movetime,
    )


def expected_score(rating: float, level: float) -> float:
    """The Elo model's expected score of a ``rating`` against a ``level``."""
    return 1.0 / (1.0 + 10.0 ** ((level - rating) / 400.0))


def _softplus(x: float) -> float:
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def _log_likelihood(rating: float, games: Sequence[tuple[float, float, int]]) -> float:
    """Of (level, points, games) results; a draw counts as half a win."""
    total = 0.0
    for level, points, count in games:
        d = (rating - level) * _LN10_400
        total -= points * _softplus(-d) + (count - points) * _softplus(d)
    return total


def _root(f: Callable[[float], float], low: float, high: float) -> float | None:
    """The root of an increasing ``f`` on [low, high] (to 0.01), None if it lies outside."""
    if f(low) > 0 or f(high) < 0:
        return None
    while high - low > 0.01:
        middle = (low + high) / 2
        low, high = (middle, high) if f(middle) < 0 else (low, middle)
    return (low + high) / 2


def ladder_rating(games: Sequence[tuple[float, float, int]]) -> dict[str, float | None]:
    """The maximum-likelihood rating for (level, points, games) results, with a 95% interval.

    ``None`` marks an unbounded end: when every game was won there is no estimate and no
    upper end (and likewise below when every game was lost).
    """
    games = [game for game in games if game[2] > 0]
    if not games:
        raise ValueError("no games")
    points = sum(game[1] for game in games)
    count = sum(game[2] for game in games)
    levels = [game[0] for game in games]
    low, high = min(levels) - 2000.0, max(levels) + 2000.0
    estimate = None
    best = 0.0  # the supremum, approached at infinity, when every game had one result
    if 0 < points < count:
        estimate = _root(
            lambda r: sum(n * expected_score(r, level) for level, _, n in games) - points, low, high
        )
        assert estimate is not None
        best = _log_likelihood(estimate, games)

    def excess(rating: float) -> float:
        return _log_likelihood(rating, games) - (best - _HALF_CHI2_95)

    lower = None if points == 0 else _root(excess, low, high if estimate is None else estimate)
    upper = None
    if points < count:
        upper = _root(lambda r: -excess(r), low if estimate is None else estimate, high)
    return {"rating": estimate, "low": lower, "high": upper}


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


def rate_ladder(entries: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """The rating fitted to ladder entries, and the bracket of levels around 50%."""
    games = [(e["level"], e["wins"] + e["draws"] / 2, e["games"]) for e in entries if e["games"]]
    if not games:
        return None
    fit = ladder_rating(games)
    played = sorted((e for e in entries if e["score"] is not None), key=lambda e: e["level"])
    above = [e["level"] for e in played if e["score"] >= 0.5]
    below = [
        e["level"] for e in played if e["score"] < 0.5 and (not above or e["level"] > above[-1])
    ]
    return {
        **{key: _rounded(value) for key, value in fit.items()},
        "games": sum(game[2] for game in games),
        "bracket": [above[-1] if above else None, below[0] if below else None],
    }


@dataclass(frozen=True)
class LadderPlan:
    weights: str
    stockfish: tuple[str, ...]
    levels: tuple[int, ...] = LADDER_LEVELS
    games: int = LADDER_GAMES
    stop_below: float = STOP_BELOW
    simulations: int = LADDER_SIMULATIONS
    scaling: tuple[int, ...] = SCALING_SIMULATIONS
    scaling_games: int = SCALING_GAMES
    movetime: float = STOCKFISH_MOVETIME


def run_ladder(
    plan: LadderPlan,
    *,
    workers: int = 1,
    device: str = "cpu",
    out: Path = Path("eval/results"),
    model_card: Path | None = None,
    report: Callable[[str], None] = print,
    watchdog: Watchdog | None = None,
    max_wall_time: float | None = None,
) -> dict[str, Any]:
    """Climb the ladder, then play the scaling curve; returns (and saves) the results.

    The results (``<checkpoint>.ladder.json``) are rewritten after every match, PGNs go to
    ``<checkpoint>.ladder/``, the chart to ``<checkpoint>.ladder.svg``. ``max_wall_time``
    (seconds) stops the run cleanly; the parts it cut off are listed in ``not_run``.
    """
    stem = Path(plan.weights).stem
    path, games_dir = out / f"{stem}.ladder.json", out / f"{stem}.ladder"
    if max_wall_time is not None:
        watchdog = replace(watchdog or Watchdog(), deadline=time.monotonic() + max_wall_time)
    deadline = watchdog.deadline if watchdog else None
    started = time.monotonic()
    results: dict[str, Any] = {
        "checkpoint": Path(plan.weights).name,
        "sha256": sha256(Path(plan.weights)),
        "suite": "ladder",
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "chessnet_version": __version__,
        "git_commit": git_commit(),
        "device": resolved_device(device),
        "hardware": hardware(device),
        "workers": workers,
        "openings": "50 balanced openings, each played with both colours",
        "scale": SCALE,
        "plan": {
            "levels": list(plan.levels),
            "games": plan.games,
            "stop_below": plan.stop_below,
            "simulations": plan.simulations,
            "scaling": list(plan.scaling),
            "scaling_games": plan.scaling_games,
            "stockfish_movetime": plan.movetime,
        },
        "ladder": [],
        "rating": None,
        "reference": None,
        "scaling": [],
        "not_climbed": [],
        "not_run": [],
        "notes": [],
    }
    out.mkdir(parents=True, exist_ok=True)
    openings = load_openings()

    def save() -> None:
        results["wall_seconds"] = round(time.monotonic() - started, 1)
        path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    def play(name: str, slug: str, player: PlayerSpec, level: int, games: int) -> dict[str, Any]:
        result = run_match(
            name, player, stockfish_at(plan.stockfish, level, plan.movetime), games=games,
            openings=openings, workers=workers, device=device, watchdog=watchdog, report=report,
        )  # fmt: skip
        pgn = write_pgn(result, games_dir / f"{slug}.pgn")
        return {
            "slug": slug, **result.summary(), "pgn": pgn.as_posix(), "level": level,
            "skill_level": round(stockfish_skill_level(level), 2),
            "device": resolved_device(device),
            "hardware": hardware(device), "workers": workers,
            "date": datetime.now(UTC).date().isoformat(), "git_commit": git_commit(),
        }  # fmt: skip

    def out_of_time() -> bool:
        return deadline is not None and time.monotonic() >= deadline

    mcts = PlayerSpec(f"mcts-{plan.simulations}", "mcts", weights=plan.weights,
                      simulations=plan.simulations)  # fmt: skip
    stopped = False
    for position, level in enumerate(plan.levels):
        if out_of_time():
            results["not_run"] += [f"ladder at UCI_Elo {lvl}" for lvl in plan.levels[position:]]
            stopped = True
            break
        entry = play(f"MCTS {plan.simulations} vs Stockfish UCI_Elo {level}", f"ladder-{level}",
                     mcts, level, plan.games)  # fmt: skip
        results["ladder"].append(entry)
        results["rating"] = rate_ladder(results["ladder"])
        save()
        if entry["incomplete"]:  # the wall-time limit cut this level short
            results["not_run"] += [
                f"ladder at UCI_Elo {lvl}" for lvl in plan.levels[position + 1 :]
            ]
            stopped = True
            break
        if entry["score"] is not None and entry["score"] < plan.stop_below:
            results["not_climbed"] = list(plan.levels[position + 1 :])
            results["notes"].append(
                f"Stopped climbing after UCI_Elo {level}: a score below {plan.stop_below:.0%}."
            )
            break

    played = [e for e in results["ladder"] if e["score"] is not None and not e["incomplete"]]
    if not played or stopped:
        results["not_run"] += [f"scaling at {n} playouts" for n in plan.scaling]
        save()
        return _finish(results, path, model_card)
    reference = min(played, key=lambda e: (abs(e["score"] - 0.5), e["level"]))
    results["reference"] = reference["level"]
    for position, simulations in enumerate(plan.scaling):
        if simulations == plan.simulations:
            entry = {**reference, "reused": "the ladder's match at this level"}
        elif out_of_time():
            results["not_run"] += [f"scaling at {n} playouts" for n in plan.scaling[position:]]
            break
        else:
            spec = PlayerSpec(f"mcts-{simulations}", "mcts", weights=plan.weights,
                              simulations=simulations)  # fmt: skip
            level = reference["level"]
            entry = play(f"MCTS {simulations} vs Stockfish UCI_Elo {level}",
                         f"scaling-{simulations}", spec, level, plan.scaling_games)  # fmt: skip
        results["scaling"].append(_rate_point(entry, simulations, reference["level"]))
        save()
        if entry["incomplete"]:
            results["not_run"] += [f"scaling at {n} playouts" for n in plan.scaling[position + 1 :]]
            break
    save()
    return _finish(results, path, model_card)


def _rate_point(entry: dict[str, Any], simulations: int, reference: int) -> dict[str, Any]:
    """A scaling point on the Stockfish scale, fitted like the ladder (one level)."""
    fit: dict[str, float | None] = {"rating": None, "low": None, "high": None}
    if entry["games"]:
        fit = ladder_rating([(reference, entry["wins"] + entry["draws"] / 2, entry["games"])])
    return {
        **entry,
        "simulations": simulations,
        "rating": _rounded(fit["rating"]),
        "rating_95": [_rounded(fit["low"]), _rounded(fit["high"])],
    }


def _finish(results: dict[str, Any], path: Path, model_card: Path | None) -> dict[str, Any]:
    chart = path.with_suffix(".svg")
    svg = scaling_chart(results)
    if svg is not None:
        chart.write_text(svg, encoding="utf-8")
        results["chart"] = chart.as_posix()
        path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if model_card is not None:
        update_model_card(model_card, results, table=partial(ladder_table, card=model_card))
    return results


def import_ladder_results(
    other: Path,
    *,
    out: Path = Path("eval/results"),
    model_card: Path | None = None,
    report: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Bring ladder results from another machine (PGNs in the directory next to them) here."""
    results = json.loads(other.read_text(encoding="utf-8"))
    if results.get("suite") != "ladder":
        raise ValueError(f"{other} holds no ladder results")
    stem = Path(results["checkpoint"]).stem
    path, games_dir = out / f"{stem}.ladder.json", out / f"{stem}.ladder"
    source = other.with_suffix("")
    for entry in [*results["ladder"], *results["scaling"]]:
        if entry.get("pgn"):
            pgn = source / Path(entry["pgn"]).name
            if not pgn.is_file():
                raise FileNotFoundError(f"the PGN of {entry['name']} is missing: {pgn}")
            games_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pgn, games_dir / pgn.name)
            entry["pgn"] = (games_dir / pgn.name).as_posix()
    results.pop("chart", None)
    # Ratings are derived from the games, so they always follow this code's method.
    results["rating"] = rate_ladder(results["ladder"])
    if results["reference"] is not None:
        results["scaling"] = [
            _rate_point(point, point["simulations"], results["reference"])
            for point in results["scaling"]
        ]
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    report(f"results: {path}")
    return _finish(results, path, model_card)


def _elo_range(low: float | None, high: float | None) -> str:
    def text(value: float | None, positive: bool) -> str:
        if value is None:
            return "+inf" if positive else "-inf"
        return format_elo(value)

    return f"{text(low, False)} to {text(high, True)}"


def _rating_text(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def _rating_range(low: float | None, high: float | None) -> str:
    if low is None and high is None:
        return "unbounded"
    if low is None:
        return f"up to {high:.0f}"
    if high is None:
        return f"{low:.0f} or more"
    return f"{low:.0f} to {high:.0f}"


def ladder_table(results: dict[str, Any], card: Path | None = None) -> str:
    """The Markdown section for a ladder run (the chart linked relative to ``card``)."""
    plan = results["plan"]
    lines = [
        f"### {results['checkpoint']}, Stockfish ladder and playout scaling",
        "",
        f"{results['created'][:10]}, {results['hardware']} ({results['device']}, "
        f"{results['workers']} worker(s)), chessnet {results['chessnet_version']}. "
        f"MCTS {plan['simulations']} against Stockfish 16 with `UCI_LimitStrength`, "
        f"{plan['stockfish_movetime']:g} s per Stockfish move; "
        f"{results['openings']}. Ratings are on {results['scale']}.",
        "",
        "| Stockfish UCI_Elo (skill level) | Games | W / D / L | Score | Elo difference (95% CI) |",
        "|---------------------------------|------:|-----------|------:|-------------------------|",
    ]
    for entry in results["ladder"]:
        name = f"{entry['level']} ({entry['skill_level']:.1f})"
        name += " (incomplete)" if entry.get("incomplete") else ""
        score = "-" if entry["score"] is None else f"{entry['score']:.1%}"
        elo = "-"
        if entry["score"] is not None:
            elo = f"{format_elo(_infinite(entry))} ({_elo_range(*entry['elo_95'])})"
        lines.append(f"| {name} | {entry['games']} | {entry['wins']} / {entry['draws']} / "
                     f"{entry['losses']} | {score} | {elo} |")  # fmt: skip
    rating = results.get("rating")
    if rating:
        lines += ["", _rating_sentence(rating, plan["simulations"])]
    if results["scaling"]:
        reference = results["reference"]
        lines += [
            "",
            f"Strength against playouts, each budget against Stockfish at UCI_Elo {reference} "
            f"(the ladder level where MCTS {plan['simulations']} scored closest to 50%):",
            "",
            f"| Playouts | Games | W / D / L | Score | Elo vs UCI_Elo {reference} | "
            "Stockfish scale (95% CI) |",
            "|---------:|------:|-----------|------:|---------------:|--------------------------|",
        ]
        for point in results["scaling"]:
            score = "-" if point["score"] is None else f"{point['score']:.1%}"
            elo = rating_cell = "-"
            if point["score"] is not None:
                elo = format_elo(_infinite(point))
                rating_cell = (f"{_rating_text(point['rating'])} "
                               f"({_rating_range(*point['rating_95'])})")  # fmt: skip
            lines.append(f"| {point['simulations']} | {point['games']} | {point['wins']} / "
                         f"{point['draws']} / {point['losses']} | {score} | {elo} | "
                         f"{rating_cell} |")  # fmt: skip
        if results.get("chart"):
            target = results["chart"]
            if card is not None:
                target = Path(os.path.relpath(results["chart"], card.parent)).as_posix()
            lines += ["", f"![Rating against playouts on Stockfish's scale]({target})"]
    reused = [p["simulations"] for p in results["scaling"] if p.get("reused")]
    notes = list(results["notes"])
    if results["scaling"]:
        notes.append("The rating and the scaling points have likelihood-ratio intervals; the "
                     "ladder table's intervals are the match tables' (normal approximation, "
                     "Wilson for a sweep).")  # fmt: skip
    if reused:
        notes.append(f"The {reused[0]}-playout point is the ladder's match at UCI_Elo "
                     f"{results['reference']}.")  # fmt: skip
    if results["not_climbed"]:
        notes.append("Not climbed: UCI_Elo " + ", ".join(map(str, results["not_climbed"])) + ".")
    errors = sum(e.get("errors", 0) for e in [*results["ladder"], *results["scaling"]])
    if errors:
        notes.append(
            f"{errors} game(s) did not finish (an error or a timeout) and are not counted."
        )
    if results["not_run"]:
        notes.append(f"Not run (wall-time limit): {'; '.join(results['not_run'])}.")
    for note in notes:
        lines += ["", note]
    return "\n".join(lines) + "\n"


def _rating_sentence(rating: dict[str, Any], simulations: int) -> str:
    """The ladder's rating in words, with its interval and its bracket."""
    head = f"**Rating of MCTS {simulations} on Stockfish's scale: "
    if rating["rating"] is not None:
        interval = _rating_range(rating["low"], rating["high"])
        head += (f"{rating['rating']:.0f} (95% CI {interval})**, the maximum-likelihood fit "
                 f"to all {rating['games']} ladder games.")  # fmt: skip
    elif rating["low"] is not None:
        head += (f"at least {rating['low']:.0f} (95%)**: it won all {rating['games']} ladder "
                 "games, so there is no point estimate.")  # fmt: skip
    else:
        head += (f"at most {rating['high']:.0f} (95%)**: it lost all {rating['games']} ladder "
                 "games, so there is no point estimate.")  # fmt: skip
    top, bottom = rating["bracket"]
    if top is not None and bottom is not None:
        return head + f" It scored 50% or more up to UCI_Elo {top}, and less from {bottom}."
    if top is not None:
        return head + f" It scored 50% or more at every level, up to UCI_Elo {top}."
    return head + f" It scored below 50% from the first level, UCI_Elo {bottom}."


def _infinite(entry: dict[str, Any]) -> float:
    """The entry's Elo, with a sweep's unbounded estimate as +-inf."""
    if entry["elo"] is not None:
        return float(entry["elo"])
    return math.inf if entry["score"] > 0.5 else -math.inf


# The chart: one series, its 95% intervals, light and dark ink from the reference palette.
_CHART_STYLE = """
  .viz { --surface: #ffffff; --ink: #191919; --ink-2: #666666; --grid: #e7e7e7;
         --axis: #c2c2c2; --series: #257de0; font-family: system-ui, -apple-system,
         "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  @media (prefers-color-scheme: dark) {
    .viz { --surface: #191919; --ink: #efefef; --ink-2: #9c9c9c; --grid: #303030;
           --axis: #474747; --series: #338ced; }
  }
  .surface { fill: var(--surface); }
  .title { fill: var(--ink); font-size: 15px; font-weight: 600; }
  .subtitle, .tick, .axis-title, .label { fill: var(--ink-2); font-size: 11.5px; }
  .grid { stroke: var(--grid); stroke-width: 1; }
  .axis { stroke: var(--axis); stroke-width: 1; }
  .line { fill: none; stroke: var(--series); stroke-width: 2; stroke-linejoin: round;
          stroke-linecap: round; }
  .whisker { stroke: var(--series); stroke-width: 2; stroke-linecap: round; opacity: 0.55; }
  .dot { fill: var(--series); stroke: var(--surface); stroke-width: 2; }
  .open { fill: var(--surface); stroke: var(--series); stroke-width: 2; }
  .hit { fill: transparent; }
"""


def scaling_chart(results: dict[str, Any]) -> str | None:
    """An SVG of rating against log2(playouts) with 95% intervals; None without points."""
    points = [p for p in results["scaling"] if p["score"] is not None]
    if not points:
        return None
    width, height = 640, 380
    left, right, top, bottom = 68, 96, 64, 56
    xs = [math.log2(p["simulations"]) for p in points]
    x_low, x_high = min(xs) - 0.4, max(xs) + 0.4
    finite = [v for p in points for v in (p["rating"], *p["rating_95"]) if v is not None]
    y_low = math.floor((min(finite) - 25) / 100) * 100
    y_high = math.ceil((max(finite) + 25) / 100) * 100
    step = 100 if y_high - y_low <= 800 else 200

    def sx(value: float) -> float:
        return left + (value - x_low) / (x_high - x_low) * (width - left - right)

    def sy(value: float) -> float:
        return top + (y_high - value) / (y_high - y_low) * (height - top - bottom)

    reference = results["reference"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" class="viz" role="img" '
        f'aria-label="Rating against playouts on Stockfish\'s scale">',
        f"<style>{_CHART_STYLE}</style>",
        f'<rect class="surface" width="{width}" height="{height}" rx="8"/>',
        f'<text class="title" x="{left}" y="26">{escape(results["checkpoint"])}: '
        "rating against playouts</text>",
        f'<text class="subtitle" x="{left}" y="45">Against Stockfish 16 at UCI_Elo {reference}. '
        "Stockfish scale, not FIDE or Lichess; 95% intervals.</text>",
    ]
    tick = y_low
    while tick <= y_high:
        y = sy(tick)
        parts.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y:.1f}" '
                     f'y2="{y:.1f}"/>')  # fmt: skip
        parts.append(f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" '
                     f'text-anchor="end">{tick}</text>')  # fmt: skip
        tick += step
    base = height - bottom
    parts.append(f'<line class="axis" x1="{left}" x2="{width - right}" y1="{base}" y2="{base}"/>')
    for point, x in zip(points, xs, strict=True):
        parts.append(f'<text class="tick" x="{sx(x):.1f}" y="{base + 18}" '
                     f'text-anchor="middle">{point["simulations"]}</text>')  # fmt: skip
    parts.append(f'<text class="axis-title" x="{(left + width - right) / 2:.1f}" '
                 f'y="{height - 12}" text-anchor="middle">'
                 "Playouts per move (log scale)</text>")  # fmt: skip
    parts.append(f'<text class="axis-title" transform="translate(18 {(top + base) / 2:.1f}) '
                 'rotate(-90)" text-anchor="middle">Rating, Stockfish scale</text>')  # fmt: skip
    if y_low <= reference <= y_high:
        y = sy(reference)
        parts.append(f'<line class="axis" x1="{left}" x2="{width - right}" y1="{y:.1f}" '
                     f'y2="{y:.1f}"/>')  # fmt: skip
        parts.append(f'<text class="label" x="{width - right + 6}" y="{y + 4:.1f}">'
                     f"UCI_Elo {reference}</text>")  # fmt: skip
    line = [(sx(x), sy(p["rating"])) for p, x in zip(points, xs, strict=True)
            if p["rating"] is not None]  # fmt: skip
    if len(line) > 1:
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{px:.1f},{py:.1f}" for i, (px, py) in enumerate(line)
        )
        parts.append(f'<path class="line" d="{path}"/>')
    for point, x in zip(points, xs, strict=True):
        cx = sx(x)
        low, high = point["rating_95"]
        y_top = sy(high) if high is not None else top
        y_bottom = sy(low) if low is not None else base
        summary = (f"{point['simulations']} playouts: {_rating_text(point['rating'])} "
                   f"({_rating_range(low, high)}), {point['games']} games, "
                   f"{point['wins']} / {point['draws']} / {point['losses']}")  # fmt: skip
        parts.append(f"<g><title>{escape(summary)}</title>")
        parts.append(f'<line class="whisker" x1="{cx:.1f}" x2="{cx:.1f}" y1="{y_top:.1f}" '
                     f'y2="{y_bottom:.1f}"/>')  # fmt: skip
        for y, bounded in ((y_top, high is not None), (y_bottom, low is not None)):
            if bounded:
                parts.append(f'<line class="whisker" x1="{cx - 5:.1f}" x2="{cx + 5:.1f}" '
                             f'y1="{y:.1f}" y2="{y:.1f}"/>')  # fmt: skip
        if point["rating"] is not None:
            parts.append(
                f'<circle class="dot" cx="{cx:.1f}" cy="{sy(point["rating"]):.1f}" r="5"/>'
            )
        else:  # a sweep: only one end is known
            known = y_bottom if low is not None else y_top
            parts.append(f'<circle class="open" cx="{cx:.1f}" cy="{known:.1f}" r="5"/>')
        parts.append(f'<rect class="hit" x="{cx - 14:.1f}" y="{top}" width="28" '
                     f'height="{base - top}"/></g>')  # fmt: skip
    for point in (points[0], points[-1]):
        if point["rating"] is not None:
            x, y = sx(math.log2(point["simulations"])), sy(point["rating"])
            anchor, dx = ("end", -10) if point is points[0] else ("start", 10)
            parts.append(f'<text class="label" x="{x + dx:.1f}" y="{y - 8:.1f}" '
                         f'text-anchor="{anchor}">{point["rating"]:.0f}</text>')  # fmt: skip
    parts.append("</svg>")
    return "\n".join(parts) + "\n"
