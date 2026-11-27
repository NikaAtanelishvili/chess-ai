"""Long evaluation runs must never hang: engines in workers, stuck engines, wall time."""

import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]


def run_script(code: str, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in its own process group, killing the whole group if it overruns."""
    process = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        pytest.fail(f"hung for more than {timeout} s:\n{output[-2000:]}")
    return subprocess.CompletedProcess(process.args, process.returncode, output)


def test_a_match_with_an_external_engine_in_worker_processes_returns(tmp_path: Path) -> None:
    """Regression: workers holding a python-chess engine never exited, so the match hung
    after its last game (the Kaggle full suite, after the Stockfish UCI_Elo match)."""
    result = run_script(
        f"""
        import sys
        sys.path.insert(0, "tests")
        from pathlib import Path
        from helpers.search import tiny_checkpoint
        from chessnet.eval.arena import run_match
        from chessnet.eval.openings import load_openings
        from chessnet.eval.players import PlayerSpec

        if __name__ == "__main__":
            weights = tiny_checkpoint(Path({str(tmp_path)!r}) / "tiny.pt")
            command = (sys.executable, "-m", "chessnet", "uci", "--weights", str(weights),
                       "--device", "cpu")
            engine = PlayerSpec("engine", "uci", command=command, options=(("Mode", "policy"),),
                                movetime=0.05)
            player = PlayerSpec("random", "random", seed=0)
            result = run_match("m", player, engine, games=2, openings=load_openings(),
                               workers=2, max_plies=20, report=print)
            print("DONE", result.stats.games)
        """,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout
    assert "DONE 2" in result.stdout


STUCK = (sys.executable, str(REPO / "tests" / "helpers" / "stuck_uci.py"))
PREAMBLE = """
import sys, time
sys.path.insert(0, "tests")
from pathlib import Path
from chessnet.eval.arena import run_match
from chessnet.eval.openings import load_openings
from chessnet.eval.players import PlayerSpec
from chessnet.eval.workers import Watchdog
RANDOM = PlayerSpec("random", "random", seed=0)
"""


def test_a_stuck_engine_call_times_out_and_the_game_counts_as_an_error() -> None:
    result = run_script(
        PREAMBLE
        + f"""
if __name__ == "__main__":
    stuck = PlayerSpec("stuck", "uci", command={STUCK!r}, movetime=0.05, timeout=1.0)
    result = run_match("m", RANDOM, stuck, games=2, openings=load_openings(),
                       watchdog=Watchdog(move_timeout=60), report=print)
    print("GAMES", result.stats.games, "ERRORS", result.summary()["errors"])
""",
        timeout=120,
    )
    assert "GAMES 0 ERRORS 2" in result.stdout, result.stdout
    assert "not finished (TimeoutError" in result.stdout  # python-chess gave up on the call


def test_a_worker_without_progress_is_killed_and_replaced_with_heartbeats() -> None:
    result = run_script(
        PREAMBLE
        + f"""
if __name__ == "__main__":
    stuck = PlayerSpec("stuck", "uci", command={STUCK!r}, nodes=1)  # no call timeout
    result = run_match("m", RANDOM, stuck, games=2, openings=load_openings(),
                       watchdog=Watchdog(move_timeout=3, heartbeat=1), report=print)
    print("GAMES", result.stats.games, "ERRORS", result.summary()["errors"])
""",
        timeout=120,
    )
    assert "GAMES 0 ERRORS 2" in result.stdout, result.stdout
    assert result.stdout.count("no progress for 3 s; the worker is killed") == 2
    assert "no progress for 3 s; the worker is killed and replaced" in result.stdout
    assert "[m] heartbeat: 0/2 done, 1 running" in result.stdout


def test_the_deadline_stops_a_match_cleanly() -> None:
    result = run_script(
        PREAMBLE
        + f"""
if __name__ == "__main__":
    stuck = PlayerSpec("stuck", "uci", command={STUCK!r}, nodes=1)
    started = time.monotonic()
    watchdog = Watchdog(move_timeout=600, deadline=time.monotonic() + 4)
    result = run_match("m", RANDOM, stuck, games=3, openings=load_openings(),
                       watchdog=watchdog, report=print)
    print("INCOMPLETE", result.incomplete, "STOPPED", len(result.failures),
          "SECONDS", round(time.monotonic() - started))
""",
        timeout=120,
    )
    assert "INCOMPLETE True STOPPED 3" in result.stdout, result.stdout
    assert int(result.stdout.split("SECONDS ")[1].split()[0]) < 30


def test_workers_that_cannot_start_end_the_match_instead_of_looping() -> None:
    result = run_script(
        PREAMBLE
        + """
if __name__ == "__main__":
    broken = PlayerSpec("broken", "uci", command=("false",), movetime=0.1)
    result = run_match("m", RANDOM, broken, games=2, openings=load_openings(),
                       watchdog=Watchdog(), report=print)
    print("ERRORS", result.summary()["errors"], result.failures[-1]["reason"][:40])
""",
        timeout=180,
    )
    assert "ERRORS 2 workers keep failing" in result.stdout, result.stdout


def test_the_suite_stops_at_its_wall_time_and_keeps_what_it_finished(tmp_path: Path) -> None:
    result = run_script(
        PREAMBLE
        + f"""
from chessnet.eval.suite import FIXTURE_PUZZLES, MatchPlan, SuitePlan, run_suite
from chessnet.eval.puzzles import read_puzzles
from helpers.search import tiny_checkpoint
import json
if __name__ == "__main__":
    out = Path({str(tmp_path)!r})
    weights = tiny_checkpoint(out / "tiny.pt")
    stuck = PlayerSpec("stuck", "uci", command={STUCK!r}, nodes=1)
    quick = MatchPlan("random vs random", "random", RANDOM, PlayerSpec("r2", "random", seed=1), 2)
    slow = MatchPlan("random vs stuck", "stuck", RANDOM, stuck, 2)
    later = MatchPlan("random again", "again", RANDOM, PlayerSpec("r3", "random", seed=2), 2)
    policy = PlayerSpec("policy", "policy", weights=str(weights))
    puzzles = list(read_puzzles(FIXTURE_PUZZLES))[:3]
    plan = SuitePlan("quick", [quick, slow, later], [policy], puzzles, "3")
    card = out / "card.md"
    results = run_suite(plan, str(weights), out=out, model_card=card, report=print,
                        watchdog=Watchdog(move_timeout=600), max_wall_time=20)
    saved = json.loads((out / "tiny.quick.json").read_text())
    matches = saved["matches"]
    print("MATCHES", [m["name"] for m in matches], [m["incomplete"] for m in matches])
    print("NOTES", saved["notes"])
    print("CARD", "(incomplete)" in card.read_text())
""",
        timeout=240,
    )
    out = result.stdout
    assert "MATCHES ['random vs random', 'random vs stuck'] [False, True]" in out, out
    assert "Not run: random again." in out
    assert "Not run: puzzles (policy)." in out
    assert "CARD True" in out
