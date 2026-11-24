import chess
import pytest

from chessnet.encoding.value import DRAW, LOSS, WIN, wdl_from_result


@pytest.mark.parametrize(
    ("result", "turn", "expected"),
    [
        ("1-0", chess.WHITE, WIN),
        ("1-0", chess.BLACK, LOSS),
        ("0-1", chess.WHITE, LOSS),
        ("0-1", chess.BLACK, WIN),
        ("1/2-1/2", chess.WHITE, DRAW),
        ("1/2-1/2", chess.BLACK, DRAW),
    ],
)
def test_wdl_is_from_the_side_to_move(result: str, turn: chess.Color, expected: int) -> None:
    assert wdl_from_result(result, turn) == expected


@pytest.mark.parametrize("result", ["*", "", "1-1", "0-0", "1/2"])
def test_unfinished_or_malformed_results_raise(result: str) -> None:
    with pytest.raises(ValueError, match="not a finished game result"):
        wdl_from_result(result, chess.WHITE)


def test_class_indices() -> None:
    assert (WIN, DRAW, LOSS) == (0, 1, 2)
