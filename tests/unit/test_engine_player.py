"""Players behind one interface: random, policy-only and MCTS."""

import chess
import pytest
import torch

from chessnet.config import MODEL_PRESETS
from chessnet.engine.player import MCTSPlayer, Player, PolicyPlayer, RandomPlayer
from chessnet.model.resnet import ChessNet
from chessnet.search.evaluator import NetworkEvaluator
from chessnet.search.mcts import SearchConfig

from helpers.search import FixedPriorEvaluator, UniformEvaluator

MATED = "R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1"
MATE_IN_ONE = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"


def test_random_player_plays_legal_moves_reproducibly() -> None:
    board = chess.Board()
    first = [RandomPlayer(seed=1).choose(board) for _ in range(3)]
    assert len(set(first)) == 1  # the same seed gives the same move
    moves = {RandomPlayer(seed=seed).choose(board) for seed in range(40)}
    assert moves <= set(board.legal_moves)
    assert len(moves) > 5


def test_policy_player_plays_the_most_likely_move() -> None:
    player = PolicyPlayer(FixedPriorEvaluator({"Nc3": 0.6}))
    assert chess.Board().san(player.choose(chess.Board())) == "Nc3"


@pytest.mark.parametrize(
    "player",
    [
        RandomPlayer(seed=0),
        PolicyPlayer(UniformEvaluator()),
        MCTSPlayer(UniformEvaluator(), simulations=10),
    ],
)
def test_players_refuse_a_finished_game(player: Player) -> None:
    with pytest.raises(ValueError, match="no legal moves"):
        player.choose(chess.Board(MATED))


def test_mcts_player_needs_a_budget() -> None:
    with pytest.raises(ValueError, match="simulations, movetime or both"):
        MCTSPlayer(UniformEvaluator())


def test_mcts_player_finds_mate_and_reports_its_search() -> None:
    board = chess.Board(MATE_IN_ONE)
    player = MCTSPlayer(UniformEvaluator(), simulations=200, config=SearchConfig(batch_size=8))
    move = player.choose(board)
    board.push(move)
    assert board.is_checkmate()
    assert player.last_result is not None
    assert player.last_result.playouts == 200
    assert player.last_result.move == move


def test_mcts_player_samples_early_moves_and_then_plays_the_most_visited() -> None:
    board = chess.Board()
    sampled = {
        MCTSPlayer(
            UniformEvaluator(), simulations=200, temperature=1.0, temperature_plies=2, seed=seed
        ).choose(board)
        for seed in range(10)
    }
    assert len(sampled) > 1  # drawn from the visit counts
    later = board.copy()
    later.push_san("e4")
    later.push_san("e5")  # ply 2: past the sampled plies
    choices = {
        MCTSPlayer(
            UniformEvaluator(), simulations=200, temperature=1.0, temperature_plies=2, seed=seed
        ).choose(later)
        for seed in range(5)
    }
    assert len(choices) == 1


def test_mcts_player_reuses_its_tree_during_a_game() -> None:
    torch.manual_seed(0)
    evaluator = NetworkEvaluator(ChessNet(MODEL_PRESETS["tiny"]))
    player = MCTSPlayer(evaluator, simulations=40, config=SearchConfig(batch_size=8))
    opponent = RandomPlayer(seed=3)
    board = chess.Board()
    reused = 0
    for _ in range(10):
        board.push(player.choose(board))
        assert player.last_result is not None
        reused += player.last_result.nodes > player.last_result.playouts
        if board.is_game_over():
            break
        board.push(opponent.choose(board))
    assert reused > 0
