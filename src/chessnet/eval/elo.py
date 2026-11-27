"""Elo differences from match results, with 95% confidence intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Two-sided 95% quantile of the normal distribution.
Z95 = 1.959964


def elo_from_score(score: float) -> float:
    """The Elo difference that predicts an expected ``score`` in (0, 1).

    ``-400 * log10(1 / score - 1)``: 0.5 is 0, 0.75 is about +191, 0.99 about +798.
    Returns +-inf for a score of 1 or 0.
    """
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"a score is between 0 and 1, not {score}")
    if score in (0.0, 1.0):
        return math.inf if score == 1.0 else -math.inf
    return -400.0 * math.log10(1.0 / score - 1.0)


def score_interval(wins: int, draws: int, losses: int, z: float = Z95) -> tuple[float, float]:
    """A 95% confidence interval for the expected score per game.

    Normal approximation with the per-game variance of win, draw and loss. When every game
    had the same result (no variance), the Wilson interval keeps the bounds informative:
    100 wins out of 100 still give a lower bound of about 0.963.
    """
    games = wins + draws + losses
    if games == 0:
        raise ValueError("no games")
    score = (wins + 0.5 * draws) / games
    variance = (wins * (1.0 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * score**2) / games
    if variance > 0.0:
        half = z * math.sqrt(variance / games)
        return max(score - half, 0.0), min(score + half, 1.0)
    centre = (score + z * z / (2 * games)) / (1 + z * z / games)
    half = z * math.sqrt(score * (1 - score) / games + z * z / (4 * games * games))
    half /= 1 + z * z / games
    # A sweep's outer bound is exactly 1 (or 0); rounding must not make it finite in Elo.
    low = 0.0 if score == 0.0 else max(centre - half, 0.0)
    high = 1.0 if score == 1.0 else min(centre + half, 1.0)
    return low, high


@dataclass(frozen=True)
class MatchStats:
    """Wins, draws and losses from one player's side, and what they imply."""

    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games

    @property
    def elo(self) -> float:
        return elo_from_score(self.score)

    @property
    def elo_interval(self) -> tuple[float, float]:
        low, high = score_interval(self.wins, self.draws, self.losses)
        return elo_from_score(low), elo_from_score(high)


def format_elo(value: float) -> str:
    """``+191``, ``-35``, or ``+inf`` / ``-inf``."""
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return f"{value:+.0f}"
