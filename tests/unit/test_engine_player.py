"""Players behind one interface: random, policy-only and MCTS."""

import chess
import pytest
import torch

from chessnet.config import MODEL_PRESETS
from chessnet.engine.player import (
    MCTSPlayer,
    Player,
    PolicyPlayer,
    PolicySafePlayer,
    RandomPlayer,
    safe_policy_move,
)
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


# --- policy-safe: the policy with three rules --------------------------------------------


class Preferring(UniformEvaluator):
    """Uniform priors, except 0.9 on one move (the policy's favourite), and a fixed value."""

    def __init__(self, favourite: chess.Move, value: float = 0.0) -> None:
        super().__init__(lambda board: value)
        self.favourite = favourite

    def evaluate(self, boards):  # type: ignore[no-untyped-def]
        evaluations = super().evaluate(boards)
        result = []
        for evaluation in evaluations:
            others = (0.1 / (len(evaluation.priors) - 1)) if len(evaluation.priors) > 1 else 0.0
            priors = {m: (0.9 if m == self.favourite else others) for m in evaluation.priors}
            result.append(type(evaluation)(priors, evaluation.value))
        return result


def mirrored(move: chess.Move) -> chess.Move:
    return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square))


def both_colours(fen: str, *moves: str) -> list[tuple[chess.Board, list[chess.Move]]]:
    """The position with White to move, and its colour mirror, with the moves translated."""
    board = chess.Board(fen)
    white = [chess.Move.from_uci(uci) for uci in moves]
    return [(board, white), (board.mirror(), [mirrored(m) for m in white])]


BETTER, LEVEL, WORSE = 0.5, 0.0, -0.5  # the value head's view of the side to move


def safe_and_plain(
    board: chess.Board, favourite: chess.Move, value: float = BETTER
) -> tuple[chess.Move, chess.Move]:
    evaluator = Preferring(favourite, value)
    return PolicySafePlayer(evaluator).choose(board), PolicyPlayer(evaluator).choose(board)


def test_policy_safe_takes_a_mate_in_one() -> None:
    for board, (favourite, mate) in both_colours(MATE_IN_ONE, "g1f1", "d1d8"):
        for value in (BETTER, LEVEL, WORSE):  # whatever the value head says
            safe, plain = safe_and_plain(board, favourite, value)
            assert safe == mate
            assert plain == favourite  # the pure policy baseline is unchanged


STALEMATE_FROM = "8/8/8/8/8/8/Q7/K1k5 w - - 0 1"  # Ka1 Qa2 against Kc1: Qe2 stalemates


def test_policy_safe_avoids_a_stalemate_when_better() -> None:
    for board, (favourite,) in both_colours(STALEMATE_FROM, "a2e2"):
        safe, plain = safe_and_plain(board, favourite, BETTER)
        assert plain == favourite
        assert safe != favourite
        board.push(safe)
        assert not board.is_stalemate()


def test_policy_safe_allows_a_stalemate_when_level_or_worse() -> None:
    """A draw is welcome when the side to move is not better (owner decision)."""
    for board, (favourite,) in both_colours(STALEMATE_FROM, "a2e2"):
        for value in (LEVEL, WORSE):
            safe, _ = safe_and_plain(board, favourite, value)
            assert safe == favourite
            board.push(safe)
            assert board.is_stalemate()
            board.pop()


SHUFFLE_FROM = "3qk3/8/8/8/8/8/8/R3K1NR w - - 0 1"
SHUFFLE = ("g1f3", "d8d7", "f3g1", "d7d8", "g1f3", "d8d7")  # back and forth twice


def shuffled(colour: chess.Color, plies: int) -> tuple[chess.Board, list[chess.Move]]:
    """The shuffle from the start position (or its mirror), ``plies`` moves in."""
    [(board, moves)] = [
        pair
        for pair in both_colours(SHUFFLE_FROM, *SHUFFLE, "f3g1", "d7d8")
        if pair[0].turn == colour
    ]
    board = board.copy()
    for move in moves[:plies]:
        board.push(move)
    return board, moves[plies:]


def test_policy_safe_does_not_complete_a_threefold_repetition_when_better() -> None:
    for colour in (chess.WHITE, chess.BLACK):
        board, rest = shuffled(colour, 7)  # the next move repeats the start a third time
        favourite = rest[0]
        board.push(favourite)
        assert board.can_claim_threefold_repetition()
        board.pop()
        safe, plain = safe_and_plain(board, favourite, BETTER)
        assert plain == favourite
        assert safe != favourite


def test_policy_safe_does_not_allow_a_threefold_claim_when_better() -> None:
    for colour in (chess.WHITE, chess.BLACK):
        board, rest = shuffled(colour, 6)  # retreating lets the opponent repeat a third time
        favourite = rest[0]
        board.push(favourite)
        assert not board.is_repetition(3)  # nothing completed yet...
        assert board.can_claim_threefold_repetition()  # ...but the reply would
        board.pop()
        safe, plain = safe_and_plain(board, favourite, BETTER)
        assert plain == favourite
        assert safe != favourite


def test_policy_safe_repeats_when_level_or_worse() -> None:
    """Completing a threefold repetition, or allowing one, is fine without an advantage."""
    for colour in (chess.WHITE, chess.BLACK):
        for plies in (7, 6):  # completes the repetition; lets the opponent complete it
            board, rest = shuffled(colour, plies)
            favourite = rest[0]
            for value in (LEVEL, WORSE):
                safe, plain = safe_and_plain(board, favourite, value)
                assert safe == plain == favourite


def test_policy_safe_plays_the_favourite_when_it_is_safe_and_falls_back_when_nothing_is() -> None:
    board = chess.Board()
    e4 = chess.Move.from_uci("e2e4")
    assert safe_and_plain(board, e4) == (e4, e4)
    stalemate = chess.Board("8/8/8/8/8/8/Q7/K1k5 w - - 0 1")
    qe2 = chess.Move.from_uci("a2e2")
    assert safe_policy_move(stalemate, {qe2: 1.0}, value=BETTER) == qe2  # the only candidate
    assert stalemate.move_stack == []  # the board is restored
