import pytest

from chessnet.config import FilterConfig
from chessnet.data.filters import HeaderFilter, Rejection, base_time, parse_headers

GOOD = {
    b"Event": b"Rated Blitz game",
    b"Site": b"https://lichess.org/game/abcdefgh",
    b"Result": b"0-1",
    b"Termination": b"Normal",
    b"TimeControl": b"180+0",
    b"WhiteElo": b"2100",
    b"BlackElo": b"2050",
}
DEFAULT = HeaderFilter(FilterConfig())


def test_good_game_passes() -> None:
    assert DEFAULT.check(GOOD) is None
    assert DEFAULT.check({**GOOD, b"Variant": b"Standard", b"SetUp": b"0"}) is None
    assert DEFAULT.check({**GOOD, b"Termination": b"Time forfeit"}) is None
    assert DEFAULT.check({**GOOD, b"Event": b"Rated Blitz tournament https://x"}) is None


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({b"Event": b"Casual Blitz game"}, Rejection.NOT_RATED),
        ({b"Variant": b"Chess960"}, Rejection.VARIANT),
        ({b"Variant": b"From Position"}, Rejection.VARIANT),
        ({b"FEN": b"8/8/8/8/8/8/8/K6k w - - 0 1"}, Rejection.VARIANT),
        ({b"SetUp": b"1"}, Rejection.VARIANT),
        ({b"Result": b"*"}, Rejection.RESULT),
        ({b"Termination": b"Abandoned"}, Rejection.TERMINATION),
        ({b"Termination": b"Rules infraction"}, Rejection.TERMINATION),
        ({b"TimeControl": b"60+0"}, Rejection.TIME_CONTROL),
        ({b"TimeControl": b"179+5"}, Rejection.TIME_CONTROL),
        ({b"TimeControl": b"-"}, Rejection.TIME_CONTROL),
        ({b"WhiteElo": b"1999"}, Rejection.ELO),
        ({b"BlackElo": b"?"}, Rejection.ELO),
        ({b"Site": b"?"}, Rejection.NO_SITE),
        ({b"Site": b""}, Rejection.NO_SITE),
    ],
)
def test_each_header_check(change: dict[bytes, bytes], reason: Rejection) -> None:
    assert DEFAULT.check({**GOOD, **change}) == reason


@pytest.mark.parametrize(
    ("missing", "reason"),
    [
        (b"Event", Rejection.NOT_RATED),
        (b"Result", Rejection.RESULT),
        (b"Termination", Rejection.TERMINATION),
        (b"TimeControl", Rejection.TIME_CONTROL),
        (b"WhiteElo", Rejection.ELO),
        (b"Site", Rejection.NO_SITE),
    ],
)
def test_missing_headers_are_rejected(missing: bytes, reason: Rejection) -> None:
    headers = {key: value for key, value in GOOD.items() if key != missing}
    assert DEFAULT.check(headers) == reason


def test_filters_are_configurable() -> None:
    no_forfeits = HeaderFilter(FilterConfig(terminations=("Normal",)))
    assert no_forfeits.check({**GOOD, b"Termination": b"Time forfeit"}) == Rejection.TERMINATION
    casual_too = HeaderFilter(FilterConfig(rated_only=False))
    assert casual_too.check({**GOOD, b"Event": b"Casual Blitz game"}) is None
    lower_floor = HeaderFilter(FilterConfig(min_elo=1800, min_base_time=60))
    assert lower_floor.check({**GOOD, b"WhiteElo": b"1800", b"TimeControl": b"60+0"}) is None


def test_unknown_ratings_fail_even_without_a_floor() -> None:
    no_floor = HeaderFilter(FilterConfig(min_elo=0))
    assert no_floor.check({**GOOD, b"WhiteElo": b"?"}) == Rejection.ELO
    assert no_floor.check({**GOOD, b"WhiteElo": b"0"}) is None


@pytest.mark.parametrize(
    ("time_control", "seconds"),
    [(b"180+2", 180), (b"600", 600), (b"0+1", 0), (b"-", None), (b"", None), (b"x+1", None)],
)
def test_base_time(time_control: bytes, seconds: int | None) -> None:
    assert base_time(time_control) == seconds


def test_parse_headers() -> None:
    block = b'[Event "Rated Blitz game"]\n[White "a \\"b\\" c"]\n[Result "1-0"]\n\n1. e4 1-0\n\n'
    assert parse_headers(block) == {
        b"Event": b"Rated Blitz game",
        b"White": b'a \\"b\\" c',
        b"Result": b"1-0",
    }


def test_parse_headers_with_crlf_line_endings() -> None:
    block = b'[Event "Rated Blitz game"]\r\n[Result "1-0"]\r\n\r\n1. e4 1-0\r\n'
    assert parse_headers(block) == {b"Event": b"Rated Blitz game", b"Result": b"1-0"}


def test_headers_without_movetext_are_malformed() -> None:
    block = b'[Event "Rated Blitz game"]\n[Result "1-0"]\n'
    assert parse_headers(block) is None
    assert DEFAULT.rejection(block) == Rejection.MALFORMED


def test_rejection_of_a_raw_block() -> None:
    lines = b"\n".join(b'[%s "%s"]' % item for item in GOOD.items())
    assert DEFAULT.rejection(lines + b"\n\n1. e4 0-1\n") is None
