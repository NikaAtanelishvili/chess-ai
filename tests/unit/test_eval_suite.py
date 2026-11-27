"""Suites: their composition, a small run end to end, the model card table, the CLI."""

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from chessnet.cli import app
from chessnet.eval.players import PlayerSpec
from chessnet.eval.puzzles import read_puzzles
from chessnet.eval.suite import (
    FIXTURE_PUZZLES,
    MatchPlan,
    SuitePlan,
    estimate_suite,
    plan_suite,
    run_suite,
    summary_table,
    update_model_card,
)

from helpers.search import tiny_checkpoint


@pytest.fixture(scope="module")
def weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny_checkpoint(tmp_path_factory.mktemp("weights") / "tiny.pt")


def test_the_quick_suite() -> None:
    plan = plan_suite("quick", "w.pt")
    assert [(m.name, m.games) for m in plan.matches] == [
        ("policy vs random", 20),
        ("MCTS 400 vs policy", 20),
    ]
    assert [p.name for p in plan.puzzle_players] == ["policy", "mcts-200"]
    assert len(plan.puzzles) == 500
    assert plan.notes == []


def test_the_full_suite_without_stockfish_or_a_second_network() -> None:
    plan = plan_suite("full", "w.pt")
    assert [m.games for m in plan.matches] == [100, 100]
    assert [p.name for p in plan.puzzle_players] == ["policy", "mcts-800"]
    assert plan.notes == ["Stockfish was not found, so its matches were skipped."]
    with pytest.raises(ValueError, match="unknown suite"):
        plan_suite("medium", "w.pt")


def test_the_full_suite_compares_a_second_network() -> None:
    plan = plan_suite("full", "base.pt", compare="other/wide.pt")
    slugs = [m.slug for m in plan.matches]
    assert slugs[-2:] == ["compare-equal-nodes", "compare-equal-time"]
    nodes, timed = plan.matches[-2:]
    assert (nodes.player.weights, nodes.player.simulations) == ("other/wide.pt", 800)
    assert nodes.opponent.weights == "base.pt"
    assert timed.player.movetime == timed.opponent.movetime == 1.0


def small_plan(weights: Path) -> SuitePlan:
    policy = PlayerSpec("policy", "policy", weights=str(weights))
    match = MatchPlan("policy vs random", "policy-vs-random", policy,
                      PlayerSpec("random", "random", seed=0), 2, "score >= 98%")  # fmt: skip
    puzzles = list(read_puzzles(FIXTURE_PUZZLES))[::100]
    return SuitePlan("quick", [match], [policy], puzzles, "5 from the fixture", ["a note"])


def test_a_small_suite_end_to_end(weights: Path, tmp_path: Path) -> None:
    card = tmp_path / "MODEL_CARD.md"
    card.write_text("# Model card\n\n## Evaluation\n", encoding="utf-8")
    results = run_suite(
        small_plan(weights), str(weights), out=tmp_path / "results", model_card=card,
        report=lambda line: None,
    )  # fmt: skip
    saved = json.loads((tmp_path / "results" / "tiny.quick.json").read_text())
    assert saved["matches"][0]["games"] == 2
    assert saved["puzzles"][0]["puzzles"] == 5
    assert saved["sha256"] == results["sha256"]
    assert Path(saved["matches"][0]["pgn"]).exists()
    text = card.read_text()
    assert "<!-- eval: tiny quick -->" in text
    assert "| policy vs random | 2 |" in text
    assert "a note" in text
    update_model_card(card, saved)  # a rerun replaces the section
    assert card.read_text().count("<!-- eval: tiny quick -->") == 1


def test_summary_table_shows_unbounded_elo() -> None:
    results = {
        "checkpoint": "base-best.pt",
        "suite": "full",
        "created": "2026-11-27T10:00:00+00:00",
        "hardware": "CPU",
        "device": "cpu",
        "workers": 1,
        "chessnet_version": "0",
        "openings": "50 balanced openings",
        "notes": [],
        "puzzles": [],
        "matches": [
            {"name": "policy vs random", "games": 100, "wins": 100, "draws": 0, "losses": 0,
             "score": 1.0, "elo": None, "elo_95": [553.2, None], "target": "score >= 98%"},
        ],
    }  # fmt: skip
    table = summary_table(results)
    assert "| 100 / 0 / 0 | 100.0% | +inf (+553 to +inf) | score >= 98% |" in table


def plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_cli_eval(weights: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    bad = runner.invoke(app, ["eval", "--suite", "medium", "--weights", str(weights)])
    assert bad.exit_code != 0
    assert "unknown suite" in plain(bad.output)
    results = tmp_path / "r.json"
    card = tmp_path / "card.md"
    run_suite(small_plan(weights), str(weights), out=tmp_path, report=lambda line: None)
    (tmp_path / "tiny.quick.json").rename(results)
    done = runner.invoke(app, ["eval", "--from-results", str(results), "--model-card", str(card)])
    assert done.exit_code == 0, done.output
    assert "| policy vs random |" in card.read_text()


def test_estimate_extrapolates_from_a_few_games_and_puzzles(weights: Path) -> None:
    plan = small_plan(weights)
    lines: list[str] = []
    rows = estimate_suite(plan, workers=2, report=lines.append)
    assert [label.split(" (")[0] for label, _, _ in rows] == ["policy vs random", "puzzles, policy"]
    for _, unit, total in rows:
        assert unit > 0
        assert total >= unit
    assert lines[0] == "Estimated wall time of the quick suite (2 worker(s)):"
    assert lines[-1].lstrip().startswith("total")
