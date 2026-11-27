"""Matches against Stockfish (skipped when Stockfish is not installed)."""

import shutil
from pathlib import Path

import pytest

from chessnet.eval.arena import run_match
from chessnet.eval.openings import load_openings
from chessnet.eval.players import PlayerSpec, uci_option_range

from helpers.search import tiny_checkpoint

STOCKFISH = shutil.which("stockfish")
pytestmark = pytest.mark.skipif(STOCKFISH is None, reason="Stockfish is not installed")


def test_stockfish_announces_its_strength_range() -> None:
    assert STOCKFISH is not None
    lowest, highest = uci_option_range((STOCKFISH,), "UCI_Elo")
    assert 1000 <= lowest < highest <= 4000


@pytest.mark.slow
def test_two_games_against_stockfish_at_skill_level_0(tmp_path: Path) -> None:
    assert STOCKFISH is not None
    weights = tiny_checkpoint(tmp_path / "tiny.pt")
    player = PlayerSpec("tiny-mcts", "mcts", weights=str(weights), simulations=16, leaf_batch=8)
    stockfish = PlayerSpec(
        "stockfish-skill-0", "uci", command=(STOCKFISH,),
        options=(("Skill Level", 0), ("Threads", 1)), movetime=0.02,
    )  # fmt: skip
    result = run_match(
        "tiny vs stockfish", player, stockfish, games=2, openings=load_openings(),
        max_plies=200, report=lambda line: None,
    )  # fmt: skip
    assert result.stats.games == 2
    assert all(record.termination != "illegal_move" for record in result.records)
