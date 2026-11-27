"""Suites: their composition, a small run end to end, the model card table, the CLI."""

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from chessnet.cli import app
from chessnet.eval.arena import MatchResult
from chessnet.eval.players import PlayerSpec
from chessnet.eval.puzzles import BUCKETS, read_puzzles
from chessnet.eval.suite import (
    FIXTURE_PUZZLES,
    MatchPlan,
    SuitePlan,
    estimate_suite,
    hardware,
    merge_suite_results,
    plan_suite,
    resolved_device,
    run_suite,
    summary_table,
    update_model_card,
    variant,
)
from chessnet.eval.workers import Watchdog

from helpers.search import tiny_checkpoint


@pytest.fixture(scope="module")
def weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny_checkpoint(tmp_path_factory.mktemp("weights") / "tiny.pt")


def test_the_quick_suite() -> None:
    plan = plan_suite("quick", "w.pt")
    assert [(m.name, m.games) for m in plan.matches] == [
        ("policy vs random", 20),
        ("policy-safe vs random", 20),
        ("MCTS 400 vs policy", 20),
    ]
    assert plan.matches[1].player.kind == "policy-safe"
    assert [p.name for p in plan.puzzle_players] == ["policy", "mcts-200"]
    assert len(plan.puzzles) == 500
    assert plan.notes == []


def test_the_full_suite_without_stockfish_or_a_second_network() -> None:
    plan = plan_suite("full", "w.pt")
    assert [m.games for m in plan.matches] == [100, 100, 100]
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


def test_summary_table_marks_what_did_not_finish() -> None:
    stopped = {
        "game": 2,
        "opening": "x",
        "reason": "stopped at the wall-time limit",
        "stopped": True,
    }
    timeout = {"game": 1, "opening": "x", "reason": "TimeoutError: ", "stopped": False}
    buckets = {label: {"puzzles": 0, "solved": 0, "accuracy": None} for label, _, _ in BUCKETS}
    results = {
        "checkpoint": "base-best.pt", "suite": "full", "created": "2026-11-27T10:00:00+00:00",
        "hardware": "CPU", "device": "cpu", "workers": 1, "chessnet_version": "0",
        "openings": "50 balanced openings", "notes": [],
        "matches": [
            {"name": "random vs stuck", "games": 0, "wins": 0, "draws": 0, "losses": 0,
             "score": None, "elo": None, "elo_95": [None, None], "target": None,
             "incomplete": True, "errors": 1, "failures": [stopped, timeout]},
        ],
        "puzzles": [
            {"player": {"name": "policy"}, "source": "3", "puzzles": 0, "accuracy": None,
             "buckets": buckets, "incomplete": True, "not_run": 2, "errors": 1,
             "failures": {"abc": "TimeoutError: "}},
        ],
        "not_run": ["random again", "puzzles (mcts-800)"],
    }  # fmt: skip
    table = summary_table(results)
    assert "| random vs stuck (incomplete) | 0 | 0 / 0 / 0 | - | - | - |" in table
    assert "| policy (incomplete) | - | - | - | - | - |" in table
    assert "1 game(s) of random vs stuck did not finish (TimeoutError: )" in table
    assert "1 puzzle(s) for policy did not finish (TimeoutError: )" in table
    assert "random vs stuck was stopped at the wall-time limit after 0 of 2 games." in table
    assert "Puzzles (policy) were stopped at the wall-time limit after 1 of 3." in table
    assert "Not run: random again; puzzles (mcts-800)." in table


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


def test_hardware_names_the_cpu_or_the_gpus() -> None:
    assert resolved_device("cpu") == "cpu"
    name = hardware("cpu")
    assert name
    assert name != "x86_64" or not Path("/proc/cpuinfo").exists()  # the model, not the arch


def test_only_plays_some_matches_and_merges_them_into_earlier_results(
    weights: Path, tmp_path: Path
) -> None:
    plan = small_plan(weights)
    safe = PlayerSpec("policy-safe", "policy-safe", weights=str(weights))
    plan.matches.insert(
        1, MatchPlan("policy-safe vs random", "policy-safe-vs-random", safe,
                     PlayerSpec("random", "random", seed=0), 2, "score >= 98%"),
    )  # fmt: skip
    first = run_suite(plan, str(weights), out=tmp_path, report=lambda line: None)
    first["hardware"] = "the other machine"  # as if the suite had run elsewhere
    path = tmp_path / "tiny.quick.json"
    path.write_text(json.dumps(first))
    card = tmp_path / "card.md"
    merged = run_suite(
        plan, str(weights), out=tmp_path, report=lambda line: None, model_card=card,
        only={"policy-safe-vs-random"},
    )  # fmt: skip
    assert [m["name"] for m in merged["matches"]] == ["policy vs random", "policy-safe vs random"]
    assert merged["puzzles"] == first["puzzles"]  # kept, not rerun
    assert "policy-safe vs random was played on" in card.read_text()
    assert merged["not_run"] == []
    saved = json.loads(path.read_text())
    saved["puzzles"][0]["solved"] = -1  # replaced by the rerun below
    path.write_text(json.dumps(saved))
    puzzles_only = run_suite(
        plan, str(weights), out=tmp_path, report=lambda line: None, only={"puzzles-policy"},
        model_card=card,
    )  # fmt: skip
    assert [run["solved"] >= 0 for run in puzzles_only["puzzles"]] == [True]
    assert puzzles_only["matches"] == saved["matches"]
    assert f"Puzzles (policy) ran on {hardware('cpu')} (cpu)." in card.read_text()
    with pytest.raises(ValueError, match="no match or puzzle run nonsense"):
        run_suite(plan, str(weights), out=tmp_path, only={"nonsense"}, report=lambda line: None)
    (tmp_path / "other").mkdir()
    other = tiny_checkpoint(tmp_path / "other" / "tiny.pt", seed=1)
    with pytest.raises(ValueError, match="another checkpoint"):
        run_suite(plan, str(other), out=tmp_path, only={"policy-safe-vs-random"},
                  report=lambda line: None)  # fmt: skip


def test_timed_matches_and_labelled_variants() -> None:
    plan = plan_suite("full", "base.pt", compare="other/wide.pt")
    timed = {m.slug for m in plan.matches if m.timed}
    assert timed == {"compare-equal-time"}  # a fixed playout budget does not depend on load
    [equal_time] = [m for m in plan.matches if m.slug == "compare-equal-time"]
    laptop = variant(equal_time, games=20, label="laptop CPU")
    assert laptop.name == "wide vs base, 1 s per move each (laptop CPU)"
    assert (laptop.slug, laptop.variant_of, laptop.games) == (
        "compare-equal-time-laptop-cpu", "compare-equal-time", 20,
    )  # fmt: skip
    assert variant(equal_time, games=10) == replace(equal_time, games=10)


def test_timed_matches_play_one_game_per_device(
    weights: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    played: list[tuple[str, int]] = []

    def fake_run_match(name, player, opponent, *, games, openings, workers, **kwargs):  # type: ignore[no-untyped-def]
        played.append((name, workers))
        return MatchResult(name, player, opponent)

    monkeypatch.setattr("chessnet.eval.suite.run_match", fake_run_match)
    policy = PlayerSpec("policy", "policy", weights=str(weights))
    timed = PlayerSpec("mcts-1s", "mcts", weights=str(weights), movetime=1.0)
    plan = SuitePlan("quick", [
        MatchPlan("fixed", "fixed", policy, PlayerSpec("random", "random", seed=0), 4),
        MatchPlan("timed", "timed", timed, policy, 4),
    ], [], [], "none")  # fmt: skip
    results = run_suite(plan, str(weights), out=tmp_path, workers=3, report=lambda line: None)
    assert played == [("fixed", 3), ("timed", 1)]  # one CPU: one timed game at a time
    assert [m["workers"] for m in results["matches"]] == [3, 1]


def test_a_labelled_variant_sits_next_to_its_match_and_does_not_replace_it(
    weights: Path, tmp_path: Path
) -> None:
    plan = small_plan(weights)
    plan.matches.append(
        MatchPlan("policy again", "again", PlayerSpec("policy", "policy", weights=str(weights)),
                  PlayerSpec("random", "random", seed=1), 2)
    )  # fmt: skip
    run_suite(plan, str(weights), out=tmp_path, only={"policy-vs-random"},
              report=lambda line: None)  # fmt: skip
    results = run_suite(
        plan, str(weights), out=tmp_path, only={"policy-vs-random"}, games=1,
        label="laptop CPU", report=lambda line: None,
    )  # fmt: skip
    names = [m["name"] for m in results["matches"]]
    assert names == ["policy vs random", "policy vs random (laptop CPU)"]
    variant_entry = results["matches"][1]
    assert (variant_entry["games"], variant_entry["variant_of"]) == (1, "policy-vs-random")
    assert Path(variant_entry["pgn"]).name == "policy-vs-random-laptop-cpu.pgn"
    assert results["not_run"] == ["policy again", "puzzles (policy)"]
    with pytest.raises(ValueError, match="apply to the matches chosen with only"):
        run_suite(plan, str(weights), out=tmp_path, games=1, report=lambda line: None)


def test_results_from_another_machine_are_merged_with_their_pgns(
    weights: Path, tmp_path: Path
) -> None:
    plan = small_plan(weights)
    plan.matches.append(
        MatchPlan("policy again", "again", PlayerSpec("policy", "policy", weights=str(weights)),
                  PlayerSpec("random", "random", seed=1), 2)
    )  # fmt: skip
    ours, theirs = tmp_path / "ours", tmp_path / "theirs"
    run_suite(plan, str(weights), out=ours, only={"policy-vs-random"}, report=lambda line: None)
    run_suite(plan, str(weights), out=theirs, only={"again", "puzzles-policy"},
              report=lambda line: None)  # fmt: skip
    card = tmp_path / "card.md"
    merged = merge_suite_results(
        plan, str(weights), theirs / "tiny.quick.json", out=ours, model_card=card,
        report=lambda line: None,
    )  # fmt: skip
    assert [m["name"] for m in merged["matches"]] == ["policy vs random", "policy again"]
    assert merged["matches"][1]["pgn"] == (ours / "tiny.quick" / "again.pgn").as_posix()
    assert (ours / "tiny.quick" / "again.pgn").read_text() == (
        theirs / "tiny.quick" / "again.pgn"
    ).read_text()
    assert [r["player"]["name"] for r in merged["puzzles"]] == ["policy"]
    assert merged["not_run"] == []
    assert json.loads((ours / "tiny.quick.json").read_text()) == merged
    assert "| policy again |" in card.read_text()
    (theirs / "tiny.quick" / "again.pgn").unlink()
    with pytest.raises(FileNotFoundError, match="PGN of policy again"):
        merge_suite_results(plan, str(weights), theirs / "tiny.quick.json", out=ours,
                            report=lambda line: None)  # fmt: skip
    saved = json.loads((theirs / "tiny.quick.json").read_text())
    saved["sha256"] = "0" * 64
    (theirs / "tiny.quick.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="another sha256"):
        merge_suite_results(plan, str(weights), theirs / "tiny.quick.json", out=ours,
                            report=lambda line: None)  # fmt: skip


def test_cli_eval_heartbeat_and_variant_options(
    weights: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "chessnet.eval.suite.run_suite", lambda *args, **kwargs: calls.append(kwargs)
    )
    runner = CliRunner()
    base = ["eval", "--weights", str(weights), "--out", str(tmp_path)]
    done = runner.invoke(app, [*base, "--heartbeat", "5", "--only", "policy-vs-random",
                               "--games", "3", "--label", "laptop CPU"])  # fmt: skip
    assert done.exit_code == 0, done.output
    assert calls[0]["watchdog"] == Watchdog(heartbeat=5.0)
    assert (calls[0]["games"], calls[0]["label"]) == (3, "laptop CPU")
    bad = runner.invoke(app, [*base, "--label", "laptop CPU"])
    assert bad.exit_code != 0
    assert "needs --only" in plain(bad.output)
