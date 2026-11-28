"""The arena: scheduling, whole games, adjudication, results and PGNs."""

import sys
from pathlib import Path

import chess
import chess.pgn
import pytest

from chessnet.eval.arena import GameRecord, play_game, run_match, schedule, write_pgn
from chessnet.eval.openings import Opening, load_openings
from chessnet.eval.players import PlayerSpec, UCIPlayer

from helpers.search import tiny_checkpoint

RANDOM_A = PlayerSpec("random-a", "random", seed=1)
RANDOM_B = PlayerSpec("random-b", "random", seed=2)


@pytest.fixture(scope="module")
def weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny_checkpoint(tmp_path_factory.mktemp("weights") / "tiny.pt")


def test_schedule_alternates_colours_and_cycles_openings() -> None:
    openings = load_openings()[:3]
    tasks = schedule(8, openings)
    assert [task.player_white for task in tasks] == [True, False] * 4
    assert [task.opening for task in tasks] == [openings[i // 2 % 3].name for i in range(8)]
    with pytest.raises(ValueError, match="no openings"):
        schedule(2, [])


class Scripted:
    def __init__(self, *moves: str) -> None:
        self.moves = iter(moves)

    def choose(self, board: chess.Board) -> chess.Move:
        return chess.Move.from_uci(next(self.moves))


def test_games_end_by_the_rules() -> None:
    board, result, termination, _ = play_game(
        Scripted("f2f3", "g2g4"), Scripted("e7e5", "d8h4"), chess.Board()
    )
    assert (result, termination) == ("0-1", "checkmate")
    assert board.is_checkmate()

    shuffle = Scripted(*["g1f3", "f3g1"] * 3)
    reply = Scripted(*["g8f6", "f6g8"] * 3)
    _, result, termination, _ = play_game(shuffle, reply, chess.Board())
    assert (result, termination) == ("1/2-1/2", "threefold_repetition")

    _, result, termination, _ = play_game(Scripted("e2e5"), Scripted(), chess.Board())
    assert (result, termination) == ("0-1", "illegal_move")

    random_white = PlayerSpec("w", "random", seed=0)
    from chessnet.eval.players import build_players

    white, black = build_players((random_white, RANDOM_B), "cpu")
    board, result, termination, thinking = play_game(white, black, chess.Board(), max_plies=10)
    assert (result, termination, len(board.move_stack)) == ("1/2-1/2", "move_limit", 10)
    assert thinking[chess.WHITE] >= 0


def test_points_from_the_players_side() -> None:
    record = GameRecord(0, "o", chess.STARTING_FEN, False, ("e2e4",), "0-1", "checkmate", 0, 0)
    assert record.player_points == 1.0  # the player had Black
    assert record.player_moves == 0
    drawn = GameRecord(1, "o", chess.STARTING_FEN, True, ("e2e4", "e7e5"), "1/2-1/2", "x", 0, 0)
    assert drawn.player_points == 0.5
    assert drawn.player_moves == 1


def test_two_games_of_the_tiny_network_against_random(weights: Path, tmp_path: Path) -> None:
    player = PlayerSpec("tiny-mcts", "mcts", weights=str(weights), simulations=16, leaf_batch=8)
    lines: list[str] = []
    result = run_match(
        "tiny vs random", player, RANDOM_A, games=2, openings=load_openings(),
        max_plies=120, target="none", report=lines.append,
    )  # fmt: skip
    assert len(result.records) == 2
    assert [record.player_white for record in result.records] == [True, False]
    assert len(lines) == 2
    summary = result.summary()
    assert summary["games"] == 2
    assert summary["wins"] + summary["draws"] + summary["losses"] == 2
    assert summary["player"]["simulations"] == 16
    assert summary["target"] == "none"

    pgn = write_pgn(result, tmp_path / "games" / "tiny.pgn")
    with pgn.open(encoding="utf-8") as file:
        games = [chess.pgn.read_game(file) for _ in range(2)]
    for game, record in zip(games, result.records, strict=True):
        assert game is not None
        assert game.headers["Result"] == record.result
        assert game.headers["FEN"] == record.fen
        board = game.end().board()
        assert [m.uci() for m in board.move_stack] == list(record.moves)
    assert {games[0].headers["White"], games[1].headers["Black"]} == {"tiny-mcts"}


def test_games_run_in_parallel_workers() -> None:
    openings = [Opening("start", chess.STARTING_FEN, "")]
    result = run_match(
        "random vs random", RANDOM_A, RANDOM_B, games=4, openings=openings, workers=2,
        max_plies=40, report=lambda line: None,
    )  # fmt: skip
    assert [record.index for record in result.records] == [0, 1, 2, 3]
    assert result.stats.games == 4


def test_an_external_uci_engine_plays_through_the_arena(weights: Path) -> None:
    """Our own engine stands in for Stockfish: a UCI subprocess with options and a limit."""
    command = (
        sys.executable,
        "-m",
        "chessnet",
        "uci",
        "--weights",
        str(weights),
        "--device",
        "cpu",
    )
    engine = PlayerSpec(
        "chessnet-uci", "uci", command=command, options=(("Mode", "policy"),), nodes=1
    )
    result = run_match(
        "uci vs random", engine, RANDOM_A, games=2, openings=load_openings(), max_plies=60,
        report=lambda line: None,
    )  # fmt: skip
    assert result.stats.games == 2
    assert all(record.termination != "illegal_move" for record in result.records)
    with pytest.raises(ValueError, match="nodes or movetime"):
        UCIPlayer(command, {})
