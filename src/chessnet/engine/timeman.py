"""Time management for games on a clock: how long to think about one move."""

from __future__ import annotations

#: The remaining time is spread over about this many future moves.
MOVES_HORIZON = 30
#: Share of the increment spent on each move.
INCREMENT_SHARE = 0.8
#: Seconds each move costs beyond its search budget: the last leaf batch running past the
#: deadline, the root evaluation, and protocol and GUI latency.
MOVE_OVERHEAD = 0.1


def move_time(
    remaining: float,
    increment: float = 0.0,
    *,
    moves_to_go: int | None = None,
    overhead: float = MOVE_OVERHEAD,
) -> float:
    """Seconds to search one move: ``remaining / 30 + 0.8 * increment``, less the overhead.

    The result is capped at half of what remains after the overhead, so no single move
    can use up the clock. With ``moves_to_go`` (a classical time control) the time is
    spread over at most that many moves. A result of 0 means there is nothing to spare:
    the search then evaluates the position once and plays the network's choice.
    """
    horizon = min(MOVES_HORIZON, moves_to_go) if moves_to_go else MOVES_HORIZON
    budget = remaining / max(horizon, 1) + INCREMENT_SHARE * increment - overhead
    cap = (remaining - overhead) / 2
    return max(0.0, min(budget, cap))
