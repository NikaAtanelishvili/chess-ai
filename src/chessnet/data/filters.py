"""Game filters.

Header checks run on the raw bytes of each game in the main process, so most games are
rejected before python-chess ever parses them. Checks that need the moves (the game length)
run later, in the parsing workers.

Whether a game is rated is not checked: the source is assumed to be a rated-games dump
(see :data:`RATED_SOURCE_ASSUMPTION`).
"""

from __future__ import annotations

import re
from enum import StrEnum

from chessnet.config import FilterConfig

#: Recorded in every dataset manifest.
RATED_SOURCE_ASSUMPTION = (
    "The source is a rated-games dump (lichess_db_standard_rated_*), so every game is rated "
    "and the Event header is not checked. Rated Swiss games, whose Event reads e.g. "
    "'Blitz swiss ...' without 'Rated', are kept; casual games in another source would be too."
)


class Rejection(StrEnum):
    """Why a game was not kept (counted in the dataset manifest)."""

    MALFORMED = "malformed"  # no header block or no movetext
    VARIANT = "variant"  # non-standard variant or custom start position
    RESULT = "result"  # unfinished or unknown result
    TERMINATION = "termination"  # e.g. abandoned, rules infraction
    TIME_CONTROL = "time_control"  # base time too short, or correspondence
    ELO = "elo"  # a player below the rating floor, or unrated
    NO_SITE = "no_site"  # no game URL to split on
    PARSE_ERROR = "parse_error"  # illegal or unparseable movetext
    TOO_SHORT = "too_short"  # fewer plies than required


RESULTS = frozenset({b"1-0", b"0-1", b"1/2-1/2"})
_TAG = re.compile(rb'^\[([A-Za-z0-9_]+)\s+"((?:[^"\\]|\\.)*)"\]\s*$', re.MULTILINE)
_MOVETEXT_START = re.compile(rb"\r?\n\r?\n")


def parse_headers(block: bytes) -> dict[bytes, bytes] | None:
    """Tag pairs of a raw game, or ``None`` if the header block is not followed by movetext."""
    end = _MOVETEXT_START.search(block)
    if end is None:
        return None
    return dict(_TAG.findall(block, 0, end.start()))


def base_time(time_control: bytes) -> int | None:
    """Base time in seconds of a PGN ``TimeControl`` like ``180+2``; ``None`` if there is none."""
    base, _, _ = time_control.partition(b"+")
    return int(base) if base.isdigit() else None


def _rating(value: bytes | None) -> int:
    return int(value) if value is not None and value.isdigit() else -1


class HeaderFilter:
    """Fast header checks on the raw bytes of a game."""

    def __init__(self, config: FilterConfig) -> None:
        self._config = config
        self._terminations = frozenset(t.encode() for t in config.terminations)

    def check(self, headers: dict[bytes, bytes]) -> Rejection | None:
        """The first failing check, or ``None`` if the game passes every header check."""
        config = self._config
        if (
            headers.get(b"Variant", b"Standard") != b"Standard"
            or b"FEN" in headers
            or headers.get(b"SetUp", b"0") != b"0"
        ):
            return Rejection.VARIANT
        if headers.get(b"Result") not in RESULTS:
            return Rejection.RESULT
        if headers.get(b"Termination") not in self._terminations:
            return Rejection.TERMINATION
        seconds = base_time(headers.get(b"TimeControl", b"-"))
        if seconds is None or seconds < config.min_base_time:
            return Rejection.TIME_CONTROL
        # Unknown ratings ("?") count as -1, so they fail even with min_elo = 0.
        if (
            min(_rating(headers.get(b"WhiteElo")), _rating(headers.get(b"BlackElo")))
            < config.min_elo
        ):
            return Rejection.ELO
        if headers.get(b"Site", b"?") in (b"", b"?"):
            return Rejection.NO_SITE
        return None

    def rejection(self, block: bytes) -> Rejection | None:
        """Parse the headers of a raw game and check them."""
        headers = parse_headers(block)
        return Rejection.MALFORMED if headers is None else self.check(headers)
