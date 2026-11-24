"""Value targets: game outcome as win/draw/loss from the side to move's perspective.

The value head predicts three classes (``WIN``, ``DRAW``, ``LOSS``) for the player to move.
Search uses the scalar ``v = P(win) - P(loss)`` in ``[-1, 1]``.
"""

from __future__ import annotations

from typing import Final

import chess

WIN: Final = 0
DRAW: Final = 1
LOSS: Final = 2
WDL_CLASSES: Final = (WIN, DRAW, LOSS)


def wdl_from_result(result: str, turn: chess.Color) -> int:
    """WDL class of a finished game's ``result`` for the player whose ``turn`` it is.

    ``result`` is a PGN result: ``"1-0"``, ``"0-1"`` or ``"1/2-1/2"``.

    Raises:
        ValueError: for any other result, including ``"*"`` (unfinished game).
    """
    if result == "1/2-1/2":
        return DRAW
    if result == "1-0":
        winner = chess.WHITE
    elif result == "0-1":
        winner = chess.BLACK
    else:
        raise ValueError(f"not a finished game result: {result!r}")
    return WIN if turn == winner else LOSS
