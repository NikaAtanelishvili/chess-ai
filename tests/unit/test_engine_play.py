"""The terminal game: board drawing, move input, and whole scripted games."""

from collections.abc import Iterator
from pathlib import Path

import chess
import pytest
from typer.testing import CliRunner

from chessnet.cli import app
from chessnet.engine.play import parse_move, play_game, render, result_text

from helpers.search import tiny_checkpoint

SCHOLAR = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"


class Scripted:
    """An engine that plays the given UCI moves in turn."""

    def __init__(self, *moves: str) -> None:
        self.moves: Iterator[str] = iter(moves)

    def choose(self, board: chess.Board) -> chess.Move:
        move = chess.Move.from_uci(next(self.moves))
        assert move in board.legal_moves
        return move


def run(engine: Scripted, inputs: list[str], human: chess.Color = chess.WHITE, **kwargs: object):  # type: ignore[no-untyped-def]
    output: list[str] = []
    feed = iter(inputs)

    def read(prompt: str) -> str:
        output.append(prompt)
        try:
            return next(feed)
        except StopIteration:
            raise EOFError from None

    board = play_game(engine, human, read=read, write=output.append, **kwargs)  # type: ignore[arg-type]
    return board, "\n".join(output)


def test_render_shows_the_board_from_either_side() -> None:
    white = render(chess.Board()).splitlines()
    assert white[1] == "8 | r n b q k b n r |"
    assert white[8] == "1 | R N B Q K B N R |"
    assert white[-1] == "    a b c d e f g h"
    black = render(chess.Board(), chess.BLACK).splitlines()
    assert black[1] == "1 | R N B K Q B N R |"
    assert black[-1] == "    h g f e d c b a"


def test_parse_move_reads_san_and_uci() -> None:
    board = chess.Board()
    assert parse_move(board, "Nf3") == parse_move(board, "g1f3") == chess.Move.from_uci("g1f3")
    with pytest.raises(ValueError, match="not legal"):
        parse_move(board, "e2e5")
    with pytest.raises(ValueError, match="not a move"):
        parse_move(board, "castle!")


def test_a_game_with_bad_input_help_undo_and_resignation() -> None:
    engine = Scripted("e7e5", "b8c6", "g8f6")
    board, output = run(engine, ["e4", "Ke9", "moves", "Nf3", "undo", "d4", "resign"])
    assert "is not a move" in output
    assert "Legal moves:" in output
    assert "chessnet plays e5" in output
    # Nf3 Nc6 were taken back; the engine then answers d4 with its next move.
    assert [move.uci() for move in board.move_stack] == ["e2e4", "e7e5", "d2d4", "g8f6"]
    assert output.endswith("You resigned.")


def test_end_of_input_resigns_and_the_engine_can_open() -> None:
    board, output = run(Scripted("e2e4"), [], human=chess.BLACK)
    assert output.splitlines()[1] == "chessnet plays e4"
    assert "You resigned." in output
    assert board.move_stack == [chess.Move.from_uci("e2e4")]


def test_the_game_ends_with_a_result() -> None:
    board, output = run(Scripted(), ["Qxf7#"], board=chess.Board(SCHOLAR))
    assert board.is_checkmate()
    assert output.endswith("You win (checkmate), 1-0.")
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert result_text(stalemate, chess.WHITE) == "Draw (stalemate), 1/2-1/2."


def test_describe_follows_each_engine_move() -> None:
    _, output = run(Scripted("e7e5"), ["e4", "resign"], describe=lambda: "eval +0.10")
    assert "chessnet plays e5 (eval +0.10)" in output


@pytest.fixture(scope="module")
def weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny_checkpoint(tmp_path_factory.mktemp("weights") / "tiny.pt")


def test_cli_play(weights: Path) -> None:
    runner = CliRunner()
    common = ["play", "--weights", str(weights), "--device", "cpu"]
    result = runner.invoke(app, [*common, "--mode", "policy"], input="e4\nresign\n")
    assert result.exit_code == 0, result.output
    assert "chessnet plays" in result.output
    assert "You resigned." in result.output
    result = runner.invoke(app, [*common, "--color", "black", "--simulations", "20"], input="")
    assert result.exit_code == 0, result.output
    assert "for White, 20 playouts" in result.output
    bad = runner.invoke(app, ["play", "--weights", str(weights.with_name("none.pt"))])
    assert bad.exit_code != 0
    assert "no checkpoint" in bad.output


def test_cli_uci(weights: Path) -> None:
    commands = "uci\nisready\nposition startpos\ngo nodes 10\n"
    result = CliRunner().invoke(app, ["uci", "--weights", str(weights)], input=commands)
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert "uciok" in lines
    assert "readyok" in lines
    assert lines[-1].startswith("bestmove ")


def test_cli_play_refuses_unknown_choices(weights: Path) -> None:
    runner = CliRunner()
    for option, value in (("--color", "green"), ("--mode", "alphabeta")):
        result = runner.invoke(app, ["play", "--weights", str(weights), option, value])
        assert result.exit_code != 0
        assert option in result.output


@pytest.mark.slow
def test_cli_export_onnx(weights: Path, tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    out = tmp_path / "tiny.onnx"
    result = CliRunner().invoke(app, ["export-onnx", "--weights", str(weights), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.stat().st_size > 100_000
