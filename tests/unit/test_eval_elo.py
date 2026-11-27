"""Elo from match scores, and its 95% confidence interval."""

import math

import pytest

from chessnet.eval.elo import MatchStats, elo_from_score, format_elo, score_interval


def test_elo_from_score() -> None:
    assert elo_from_score(0.5) == 0.0
    assert elo_from_score(0.75) == pytest.approx(190.85, abs=0.01)
    assert elo_from_score(0.25) == pytest.approx(-190.85, abs=0.01)
    assert elo_from_score(0.99) == pytest.approx(798.25, abs=0.01)
    assert elo_from_score(1.0) == math.inf
    assert elo_from_score(0.0) == -math.inf
    scores = [elo_from_score(s / 100) for s in range(1, 100)]
    assert scores == sorted(scores)
    with pytest.raises(ValueError, match="between 0 and 1"):
        elo_from_score(1.5)


def test_score_interval_contains_the_score_and_narrows_with_games() -> None:
    low, high = score_interval(60, 20, 20)
    assert low < 0.7 < high
    wide = score_interval(6, 2, 2)
    assert wide[1] - wide[0] > high - low
    assert score_interval(20, 20, 60) == pytest.approx((1 - high, 1 - low))  # symmetric


def test_a_clean_sweep_still_has_a_finite_lower_bound() -> None:
    low, high = score_interval(100, 0, 0)
    assert high == 1.0
    assert 0.95 < low < 0.97  # Wilson: n / (n + z**2) for a perfect score
    assert score_interval(0, 0, 100)[1] < 0.05
    assert score_interval(0, 50, 0) == pytest.approx(
        (score_interval(0, 50, 0)[0], score_interval(0, 50, 0)[1])
    )


def test_match_stats() -> None:
    stats = MatchStats(wins=7, draws=2, losses=1)
    assert stats.games == 10
    assert stats.score == 0.8
    assert stats.elo == pytest.approx(240.8, abs=0.1)
    low, high = stats.elo_interval
    assert low < stats.elo < high
    sweep = MatchStats(10, 0, 0)
    assert sweep.elo == math.inf
    assert math.isfinite(sweep.elo_interval[0])
    assert format_elo(sweep.elo) == "+inf"
    assert format_elo(-35.4) == "-35"
    with pytest.raises(ValueError, match="no games"):
        score_interval(0, 0, 0)
