"""Whole games: MCTS with the tiny network against a random mover, legal to the end."""

import chess
import pytest
import torch

from chessnet.config import MODEL_PRESETS
from chessnet.engine.player import MCTSPlayer, Player, RandomPlayer
from chessnet.model.resnet import ChessNet
from chessnet.search.evaluator import NetworkEvaluator
from chessnet.search.mcts import SearchConfig


def play(white: Player, black: Player, *, max_plies: int = 2000) -> chess.Board:
    """Play until the game is over, draws by claim included; every move must be legal."""
    board = chess.Board()
    while board.outcome(claim_draw=True) is None:
        assert board.ply() < max_plies
        move = (white if board.turn == chess.WHITE else black).choose(board)
        assert move in board.legal_moves
        board.push(move)
    return board


def mcts_player(evaluator: NetworkEvaluator) -> MCTSPlayer:
    return MCTSPlayer(evaluator, simulations=16, config=SearchConfig(batch_size=8))


@pytest.fixture(scope="module")
def evaluator() -> NetworkEvaluator:
    torch.manual_seed(0)
    return NetworkEvaluator(ChessNet(MODEL_PRESETS["tiny"]), cache_size=20_000)


def test_one_game_against_a_random_mover(evaluator: NetworkEvaluator) -> None:
    board = play(mcts_player(evaluator), RandomPlayer(seed=0))
    assert board.outcome(claim_draw=True) is not None


@pytest.mark.slow
def test_fifty_games_against_a_random_mover_stay_legal(evaluator: NetworkEvaluator) -> None:
    terminations = []
    for game in range(50):
        mcts, random_player = mcts_player(evaluator), RandomPlayer(seed=game)
        white, black = (mcts, random_player) if game % 2 == 0 else (random_player, mcts)
        outcome = play(white, black).outcome(claim_draw=True)
        assert outcome is not None
        terminations.append(outcome.termination)
    assert len(terminations) == 50
