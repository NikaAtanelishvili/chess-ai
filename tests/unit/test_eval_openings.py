"""The opening book: 50 distinct, legal, balanced positions."""

import re
from importlib import resources

import chess

from chessnet.eval.openings import load_openings


def test_the_book_has_fifty_distinct_legal_openings() -> None:
    openings = load_openings()
    assert len(openings) == 50
    assert len({opening.fen for opening in openings}) == 50
    assert len({opening.name for opening in openings}) == 50
    for opening in openings:
        board = chess.Board()
        for san in opening.line.split():
            board.push_san(san)  # the recorded line leads to the position
        assert board.epd() == opening.board().epd()
        assert opening.board().turn == chess.WHITE
        assert opening.board().is_valid()


def test_every_opening_is_balanced() -> None:
    text = resources.files("chessnet.eval").joinpath("data/openings.epd").read_text()
    scores = [int(score) for score in re.findall(r"\bce (-?\d+);", text)]
    assert len(scores) == 50
    assert max(abs(score) for score in scores) <= 60  # Stockfish 16, depth 20


def test_a_custom_book(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "book.epd"
    path.write_text(f'{chess.Board().epd()} id "start";\n\n', encoding="utf-8")
    [opening] = load_openings(path)
    assert opening.name == "start"
    assert opening.board() == chess.Board()
