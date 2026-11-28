"""The Stockfish ladder: the rating fit, climbing and stopping, scaling, the chart, the CLI."""

import json
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Any

import chess
import pytest
from typer.testing import CliRunner

from chessnet.cli import app
from chessnet.eval.arena import GameRecord, MatchResult
from chessnet.eval.ladder import (
    LadderPlan,
    expected_score,
    import_ladder_results,
    ladder_rating,
    rate_ladder,
    run_ladder,
    scaling_chart,
    stockfish_skill_level,
)
from chessnet.eval.workers import Watchdog

from helpers.search import tiny_checkpoint


@pytest.fixture(scope="module")
def weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny_checkpoint(tmp_path_factory.mktemp("weights") / "tiny.pt")


def test_stockfish_16_turns_uci_elo_into_a_skill_level() -> None:
    assert stockfish_skill_level(1320) == 0.0  # the same opponent as Skill Level 0
    assert stockfish_skill_level(1500) == pytest.approx(1.489, abs=0.001)
    assert stockfish_skill_level(2300) == pytest.approx(5.513, abs=0.001)


def test_one_level_gives_the_elo_formula_with_an_interval() -> None:
    assert expected_score(1900, 1900) == 0.5
    even = ladder_rating([(1900, 20, 40)])
    assert even["rating"] == pytest.approx(1900, abs=0.05)
    fit = ladder_rating([(1900, 30, 40)])  # 75%
    assert fit["rating"] == pytest.approx(1900 + 190.85, abs=0.05)
    assert fit["low"] < fit["rating"] < fit["high"]
    narrow = ladder_rating([(1900, 300, 400)])
    assert narrow["high"] - narrow["low"] < fit["high"] - fit["low"]  # more games, tighter


def test_several_levels_are_fitted_together() -> None:
    fit = ladder_rating([(1700, 30, 40), (1900, 18, 40), (2100, 6, 40)])
    assert 1850 < fit["rating"] < 1950
    single = ladder_rating([(1900, 18, 40)])
    assert fit["high"] - fit["low"] < single["high"] - single["low"]  # all games count


def test_a_sweep_has_one_bound_and_no_estimate() -> None:
    won = ladder_rating([(1500, 40, 40)])
    assert won["rating"] is None
    assert won["high"] is None
    assert 1900 < won["low"] < 2200
    lost = ladder_rating([(1500, 0, 40)])
    assert lost["rating"] is None
    assert lost["low"] is None
    assert 800 < lost["high"] < 1100
    with pytest.raises(ValueError, match="no games"):
        ladder_rating([])


def entry(level: int, wins: int, draws: int, losses: int) -> dict[str, Any]:
    games = wins + draws + losses
    return {"level": level, "wins": wins, "draws": draws, "losses": losses, "games": games,
            "score": (wins + draws / 2) / games}  # fmt: skip


def test_the_bracket_is_where_the_score_crosses_50_percent() -> None:
    rating = rate_ladder([entry(1500, 35, 2, 3), entry(1700, 22, 6, 12), entry(1900, 10, 6, 24)])
    assert rating is not None
    assert rating["bracket"] == [1700, 1900]
    assert rating["games"] == 120
    top = rate_ladder([entry(1500, 40, 0, 0)])
    assert top is not None
    assert top["bracket"] == [1500, None]
    assert top["rating"] is None


def fake_match(scores: dict[tuple[int, int], float], calls: list[tuple[int, int, int]]):  # type: ignore[no-untyped-def]
    """A run_match stand-in: wins and losses by (playouts, UCI_Elo), no draws."""

    def run_match(name, player, opponent, *, games, **kwargs):  # type: ignore[no-untyped-def]
        level = dict(opponent.options)["UCI_Elo"]
        calls.append((player.simulations, level, games))
        wins = round(scores[(player.simulations, level)] * games)
        records = []
        for index in range(games):
            white = index % 2 == 0
            won = index < wins
            result = "1-0" if white == won else "0-1"
            records.append(GameRecord(index, "start", chess.STARTING_FEN, white, (), result,
                                      "checkmate", 0.0, 0.0))  # fmt: skip
        return MatchResult(name, player, opponent, records=records, seconds=1.0)

    return run_match


SCORES = {(800, 1500): 0.9, (800, 1700): 0.7, (800, 1900): 0.45, (800, 2100): 0.2, (800, 2300): 0.0,
          (50, 1900): 0.0, (100, 1900): 0.1, (200, 1900): 0.2, (400, 1900): 0.35,
          (1600, 1900): 0.6}  # fmt: skip


def test_the_ladder_stops_climbing_and_scales_against_the_level_nearest_50_percent(
    weights: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, int, int]] = []
    monkeypatch.setattr("chessnet.eval.ladder.run_match", fake_match(SCORES, calls))
    card = tmp_path / "card.md"
    card.write_text("# Model card\n", encoding="utf-8")
    plan = LadderPlan(str(weights), ("stockfish",), games=20, scaling_games=10)
    results = run_ladder(plan, out=tmp_path / "results", model_card=card, report=lambda line: None)
    assert [e["level"] for e in results["ladder"]] == [1500, 1700, 1900, 2100]
    assert results["not_climbed"] == [2300]
    assert results["reference"] == 1900  # 45% is the closest to 50%
    assert calls[4:] == [(50, 1900, 10), (100, 1900, 10), (200, 1900, 10), (400, 1900, 10),
                         (1600, 1900, 10)]  # the 800 point is the ladder's match  # fmt: skip
    points = {p["simulations"]: p for p in results["scaling"]}
    assert list(points) == [50, 100, 200, 400, 800, 1600]
    assert points[800]["reused"]
    assert points[800]["games"] == 20
    assert points[50]["rating"] is None
    assert points[50]["rating_95"][0] is None
    assert points[1600]["rating"] == pytest.approx(1900 + 70.4, abs=0.1)
    rating = results["rating"]
    assert rating["bracket"] == [1700, 1900]
    assert 1800 < rating["rating"] < 1900
    saved = json.loads((tmp_path / "results" / "tiny.ladder.json").read_text())
    assert saved["reference"] == 1900
    assert saved["chart"].endswith("tiny.ladder.svg")
    ET.fromstring((tmp_path / "results" / "tiny.ladder.svg").read_text())  # well-formed SVG
    text = card.read_text()
    assert "<!-- eval: tiny ladder -->" in text
    assert "**Rating of MCTS 800 on Stockfish's scale:" in text
    assert "| 1900 (3.8) | 20 | 9 / 0 / 11 | 45.0% |" in text
    assert "![Rating against playouts on Stockfish's scale](results/tiny.ladder.svg)" in text
    assert "Not climbed: UCI_Elo 2300." in text
    assert "Elo per doubling of the playouts (95% CI" in text
    assert results["scaling_fit"]["elo_per_doubling"] > 0
    assert "not a FIDE or Lichess one" in text


def test_the_wall_time_limit_stops_the_ladder_before_the_next_match(
    weights: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, int, int]] = []
    monkeypatch.setattr("chessnet.eval.ladder.run_match", fake_match(SCORES, calls))
    plan = LadderPlan(str(weights), ("stockfish",), games=4, scaling=(50, 800))
    expired = Watchdog(deadline=time.monotonic() - 1)
    results = run_ladder(plan, out=tmp_path, watchdog=expired, report=lambda line: None)
    assert calls == []
    assert results["not_run"][0] == "ladder at UCI_Elo 1500"
    assert results["not_run"][-2:] == ["scaling at 50 playouts", "scaling at 800 playouts"]
    assert "chart" not in results


def test_results_from_another_machine_are_imported_with_their_pgns(
    weights: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("chessnet.eval.ladder.run_match", fake_match(SCORES, []))
    plan = LadderPlan(str(weights), ("stockfish",), games=4, scaling=(100, 800), scaling_games=4)
    run_ladder(plan, out=tmp_path / "kaggle", report=lambda line: None)
    card = tmp_path / "docs" / "card.md"
    card.parent.mkdir()
    out = tmp_path / "eval"
    results = import_ladder_results(tmp_path / "kaggle" / "tiny.ladder.json", out=out,
                                    model_card=card, report=lambda line: None)  # fmt: skip
    assert results["ladder"][0]["pgn"] == (out / "tiny.ladder" / "ladder-1500.pgn").as_posix()
    assert results["rating"] == rate_ladder(results["ladder"])  # derived again, from the games
    assert (out / "tiny.ladder" / "scaling-100.pgn").exists()
    assert (out / "tiny.ladder.svg").exists()
    assert "](../eval/tiny.ladder.svg)" in card.read_text()
    (tmp_path / "kaggle" / "tiny.ladder" / "scaling-100.pgn").unlink()
    with pytest.raises(FileNotFoundError, match="PGN of MCTS 100"):
        import_ladder_results(tmp_path / "kaggle" / "tiny.ladder.json", out=out,
                              report=lambda line: None)  # fmt: skip


def test_the_chart_shows_unbounded_points_by_their_known_end() -> None:
    def point(simulations: int, rating: float | None, low: float | None, high: float | None):  # type: ignore[no-untyped-def]
        return {"simulations": simulations, "score": 0.5 if rating else 0.0, "rating": rating,
                "rating_95": [low, high], "games": 30, "wins": 15, "draws": 0,
                "losses": 15}  # fmt: skip

    results = {"checkpoint": "base-best.pt", "reference": 1900,
               "scaling": [point(50, None, None, 1500.0), point(400, 1850.0, 1780.0, 1920.0),
                           point(1600, 2010.0, 1930.0, 2090.0)]}  # fmt: skip
    svg = scaling_chart(results)
    assert svg is not None
    root = ET.fromstring(svg)
    circles = list(root.iter("{http://www.w3.org/2000/svg}circle"))
    assert sorted(c.get("class") for c in circles) == ["dot", "dot", "open"]
    titles = [t.text for t in root.iter("{http://www.w3.org/2000/svg}title")]
    assert "50 playouts: - (up to 1500), 30 games, 15 / 0 / 15" in titles
    assert "prefers-color-scheme: dark" in svg
    assert scaling_chart({**results, "scaling": []}) is None
    assert math.isclose(expected_score(2000, 1900), 0.640, abs_tol=0.001)


def test_cli_ladder(weights: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plans: list[LadderPlan] = []
    monkeypatch.setattr("chessnet.eval.ladder.run_ladder",
                        lambda plan, **kwargs: plans.append(plan) or {})  # fmt: skip
    runner = CliRunner()
    base = ["ladder", "--weights", str(weights), "--stockfish", "sf", "--out", str(tmp_path)]
    done = runner.invoke(app, [*base, "--levels", "1700,1500", "--scaling", "", "--games", "10"])
    assert done.exit_code == 0, done.output
    assert (plans[0].levels, plans[0].scaling, plans[0].games) == ((1500, 1700), (), 10)
    bad = runner.invoke(app, [*base, "--levels", "1500,high"])
    assert bad.exit_code != 0


def test_a_sweep_on_the_ladder_is_reported_as_a_bound() -> None:
    from chessnet.eval.ladder import _rating_sentence

    won = rate_ladder([entry(1500, 40, 0, 0)])
    assert won is not None
    sentence = _rating_sentence(won, 800)
    assert "at least" in sentence
    assert "no point estimate" in sentence
    assert "50% or more at every level, up to UCI_Elo 1500." in sentence
    lost = rate_ladder([entry(1500, 0, 0, 40)])
    assert lost is not None
    assert "below 50% from the first level, UCI_Elo 1500." in _rating_sentence(lost, 800)


def curve_points(rating, budgets=(50, 100, 200, 400, 800, 1600), games=2000):  # type: ignore[no-untyped-def]
    """Scaling points whose scores against UCI_Elo 2300 follow ``rating(playouts)`` exactly."""
    return [{"simulations": n, "games": games, "wins": games * expected_score(rating(n), 2300),
             "draws": 0, "losses": 0} for n in budgets]  # fmt: skip


def test_the_scaling_curve_is_fitted_as_a_line_in_log2_playouts() -> None:
    from chessnet.eval.ladder import scaling_fit

    fit = scaling_fit(curve_points(lambda n: 2400 + 100 * math.log2(n / 400)), 2300)
    assert fit is not None
    assert fit["elo_per_doubling"] == pytest.approx(100, abs=0.5)
    assert fit["low"] < 100 < fit["high"]
    assert fit["curvature_p"] > 0.9
    assert fit["fit_p"] > 0.9
    assert [n for n, _ in fit["line"]] == [50, 100, 200, 400, 800, 1600]
    assert fit["line"][3][1] == pytest.approx(2400, abs=0.5)
    json.dumps(fit)  # plain numbers, ready for the results file
    bent = scaling_fit(curve_points(lambda n: 2400 + 100 * math.log2(n / 400)
                                    - 40 * math.log2(n / 400) ** 2), 2300)  # fmt: skip
    assert bent is not None
    assert bent["curvature_p"] < 0.01  # flattening is detected
    assert scaling_fit(curve_points(lambda n: 2400.0, budgets=(100, 400)), 2300) is None


def test_a_named_ladder_has_its_own_results_and_section(
    weights: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("chessnet.eval.ladder.run_match", fake_match({(1600, 1700): 0.6}, []))
    card = tmp_path / "card.md"
    plan = LadderPlan(str(weights), ("sf",), name="ladder-1600", levels=(1700,), games=10,
                      simulations=1600, scaling=())  # fmt: skip
    results = run_ladder(plan, out=tmp_path, model_card=card, report=lambda line: None)
    assert results["suite"] == "ladder-1600"
    assert (tmp_path / "tiny.ladder-1600.json").exists()
    assert (tmp_path / "tiny.ladder-1600" / "ladder-1700.pgn").exists()
    text = card.read_text()
    assert "<!-- eval: tiny ladder-1600 -->" in text
    assert "### tiny.pt, Stockfish ladder, MCTS 1600\n" in text
    with pytest.raises(ValueError, match="a ladder's name"):
        run_ladder(replace(plan, name="../oops"), out=tmp_path, report=lambda line: None)


def test_an_extension_is_merged_into_the_ladder_it_continues(
    weights: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scores = {(800, 1500): 0.9, (800, 1700): 0.8, (800, 1900): 0.6, (800, 2100): 0.4,
              (100, 1700): 0.5}  # fmt: skip
    monkeypatch.setattr("chessnet.eval.ladder.run_match", fake_match(scores, []))
    ours, kaggle = tmp_path / "eval", tmp_path / "kaggle"
    first = LadderPlan(str(weights), ("sf",), levels=(1500, 1700), games=10, scaling=(100,),
                       scaling_games=10)  # fmt: skip
    run_ladder(first, out=ours, report=lambda line: None)
    before = json.loads((ours / "tiny.ladder.json").read_text())
    more = replace(first, levels=(1900, 2100), scaling=())
    run_ladder(more, out=kaggle, report=lambda line: None)
    merged = import_ladder_results(kaggle / "tiny.ladder.json", out=ours, merge=True,
                                   report=lambda line: None)  # fmt: skip
    assert [e["level"] for e in merged["ladder"]] == [1500, 1700, 1900, 2100]
    assert merged["scaling"] == before["scaling"]  # the curve stays
    assert merged["reference"] == before["reference"] == 1700
    assert merged["rating"] == rate_ladder(merged["ladder"])
    assert merged["rating"]["bracket"] == [1900, 2100]
    assert merged["plan"]["levels"] == [1500, 1700, 1900, 2100]
    assert [run["levels"] for run in merged["runs"]] == [[1500, 1700], [1900, 2100]]
    assert (ours / "tiny.ladder" / "ladder-2100.pgn").exists()
    other = replace(more, simulations=1600, name="ladder")
    monkeypatch.setattr("chessnet.eval.ladder.run_match",
                        fake_match({(1600, 1900): 0.5, (1600, 2100): 0.5}, []))  # fmt: skip
    run_ladder(other, out=kaggle, report=lambda line: None)
    with pytest.raises(ValueError, match="other playouts"):
        import_ladder_results(kaggle / "tiny.ladder.json", out=ours, merge=True,
                              report=lambda line: None)  # fmt: skip
