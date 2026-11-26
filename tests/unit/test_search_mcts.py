"""MCTS with stub evaluators, so that these tests check the search itself, above all its signs."""

import random
import threading
import time
from collections import Counter
from collections.abc import Iterator, Sequence

import chess
import pytest

from chessnet.search.evaluator import Evaluation
from chessnet.search.mcts import (
    MCTS,
    Node,
    SearchConfig,
    backup,
    most_visited,
    sample_move,
    terminal_value,
)

from helpers.search import FixedPriorEvaluator, MaterialEvaluator, UniformEvaluator

MATE_IN_ONE = (
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",  # back rank: Rd8#
    "k7/8/1K6/8/8/8/8/7Q w - - 0 1",  # queen and king: Qh8#
    "6rk/6pp/8/6N1/8/8/8/6K1 w - - 0 1",  # smothered: Nf7#
)
#: Seven moves let Black mate at once: the rook leaving the first rank (...Rd1#), or Rd1 (...Rxd1#).
ALLOWS_MATE = "3r2k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
HANGING = (
    "4k3/8/8/8/3n4/8/8/3RK3 w - - 0 1",  # Rxd4 wins an undefended knight
    "4k3/8/8/1r6/8/8/4B3/4K3 w - - 0 1",  # Bxb5+ wins an undefended rook
)
#: Black is four points down here, and up eleven in WINNING; White shuffles a knight.
LOSING = "3qk3/8/8/8/8/8/8/R3K1NR w - - 0 1"
WINNING = "r2qk3/8/8/8/8/8/8/4K1N1 w - - 0 1"
#: After these moves the start position has occurred twice, and ...Qd8 brings it back a third time.
SHUFFLE = ("Nf3", "Qd7", "Ng1", "Qd8", "Nf3", "Qd7", "Ng1")


def both_colours(fen: str) -> list[chess.Board]:
    """The position, and its colour mirror with the other side to move."""
    return [chess.Board(fen), chess.Board(fen).mirror()]


def after(board: chess.Board, move: chess.Move) -> chess.Board:
    board = board.copy()
    board.push(move)
    return board


def allows_mate(board: chess.Board, move: chess.Move) -> bool:
    following = after(board, move)
    return any(after(following, reply).is_checkmate() for reply in following.legal_moves)


def after_shuffle(fen: str) -> chess.Board:
    board = chess.Board(fen)
    for san in SHUFFLE:
        board.push_san(san)
    return board


def walk(node: Node) -> Iterator[Node]:
    yield node
    for child in (node.children or {}).values():
        yield from walk(child)


def search(board: chess.Board, evaluator: object, simulations: int, **config: int) -> MCTS:
    mcts = MCTS(evaluator, SearchConfig(**config))  # type: ignore[arg-type]
    mcts.search(board, simulations=simulations)
    return mcts


# --- signs -----------------------------------------------------------------------------


def test_backup_negates_the_value_at_every_ply() -> None:
    root, child, grandchild = Node(1.0), Node(0.5), Node(0.5)
    backup([root, child], 0.7)  # a leaf one ply down, worth +0.7 to its side to move
    assert child.q == pytest.approx(-0.7)  # what the root's mover gets from that child
    assert root.q == pytest.approx(0.7)
    backup([root, child, grandchild], 0.5)  # two plies down: the root's mover moves again
    assert grandchild.q == pytest.approx(-0.5)
    assert child.value_sum == pytest.approx(-0.7 + 0.5)
    assert root.value_sum == pytest.approx(0.7 - 0.5)
    assert (root.visits, child.visits, grandchild.visits) == (2, 2, 1)


@pytest.mark.parametrize(
    ("fen", "winning"),
    [("4k3/8/8/8/8/8/8/QQQ1K3 w - - 0 1", True), ("qqq1k3/8/8/8/8/8/8/4K3 w - - 0 1", False)],
)
def test_the_value_is_for_the_side_to_move(fen: str, winning: bool) -> None:
    for board in both_colours(fen):
        value = MCTS(MaterialEvaluator()).search(board, simulations=200).value
        assert value > 0.5 if winning else value < -0.5


# --- terminal positions ----------------------------------------------------------------


def test_terminal_values() -> None:
    assert terminal_value(chess.Board("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1")) == -1.0  # mated
    assert terminal_value(chess.Board("R5k1/5ppp/8/8/8/8/8/6K1 b - - 100 80")) == -1.0
    assert terminal_value(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")) == 0.0  # stalemate
    assert terminal_value(chess.Board("8/8/8/4k3/8/8/8/4K3 w - - 0 1")) == 0.0  # bare kings
    assert terminal_value(chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 100 80")) == 0.0  # 50 moves
    assert terminal_value(chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 99 80")) is None
    assert terminal_value(chess.Board()) is None


# --- mates and material ----------------------------------------------------------------


@pytest.mark.parametrize("fen", MATE_IN_ONE)
@pytest.mark.parametrize("batch_size", [1, 16])
def test_finds_mate_in_one_for_both_colours(fen: str, batch_size: int) -> None:
    for board in both_colours(fen):
        result = MCTS(UniformEvaluator(), SearchConfig(batch_size=batch_size)).search(
            board, simulations=200
        )
        assert after(board, result.move).is_checkmate()
        assert result.value > 0.5


@pytest.mark.parametrize("batch_size", [1, 16])
def test_avoids_moves_that_allow_mate_in_one(batch_size: int) -> None:
    for board in both_colours(ALLOWS_MATE):
        losing = {move for move in board.legal_moves if allows_mate(board, move)}
        assert len(losing) == 7
        result = MCTS(UniformEvaluator(), SearchConfig(batch_size=batch_size)).search(
            board, simulations=2000
        )
        assert result.move not in losing


@pytest.mark.parametrize("fen", HANGING)
def test_wins_hanging_material_for_both_colours(fen: str) -> None:
    for board in both_colours(fen):
        result = MCTS(MaterialEvaluator()).search(board, simulations=400)
        assert board.is_capture(result.move)
        assert board.piece_type_at(result.move.to_square) in {chess.KNIGHT, chess.ROOK}


def test_priors_steer_the_search() -> None:
    board = chess.Board()
    result = MCTS(FixedPriorEvaluator({"d4": 0.9})).search(board, simulations=100)
    assert board.san(result.move) == "d4"


# --- repetition: required regression test ----------------------------------------------


def test_threefold_repetition_is_detected_from_the_real_move_stack() -> None:
    """The legacy search reversed or invented the board history (docs/AUDIT.md, X4)."""
    board = after_shuffle(LOSING)
    qd8 = board.parse_san("Qd8")
    mcts = MCTS(MaterialEvaluator())
    result = mcts.search(board, simulations=800)
    assert mcts.root is not None
    assert mcts.root.children is not None
    assert mcts.root.children[qd8].terminal == 0.0  # the third occurrence: a draw
    assert result.move == qd8  # which the losing side takes

    # The same position without its history repeats nothing.
    fresh = MCTS(MaterialEvaluator())
    fresh.search(chess.Board(board.fen()), simulations=800)
    assert fresh.root is not None
    assert fresh.root.children is not None
    child = fresh.root.children[qd8]
    assert child.visits > 0
    assert child.terminal is None


def test_the_winning_side_avoids_the_repetition() -> None:
    board = after_shuffle(WINNING)
    result = MCTS(MaterialEvaluator()).search(board, simulations=800)
    assert result.move != board.parse_san("Qd8")
    assert result.value > 0.9


def test_a_root_where_a_draw_could_be_claimed_is_still_searched() -> None:
    board = after_shuffle(LOSING)
    board.push_san("Qd8")
    assert board.is_repetition(3)
    result = MCTS(UniformEvaluator()).search(board, simulations=50)
    assert result.move in board.legal_moves


def test_the_search_leaves_the_game_board_alone() -> None:
    board = after_shuffle(LOSING)
    before = (board.fen(), list(board.move_stack))
    MCTS(UniformEvaluator()).search(board, simulations=300)
    assert (board.fen(), list(board.move_stack)) == before


def test_a_finished_game_cannot_be_searched() -> None:
    with pytest.raises(ValueError, match="no legal moves"):
        MCTS(UniformEvaluator()).search(
            chess.Board("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1"), simulations=9
        )


# --- budgets, batches and statistics ---------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 3, 16, 64])
def test_simulations_are_counted_exactly(batch_size: int) -> None:
    evaluator = UniformEvaluator()
    mcts = MCTS(evaluator, SearchConfig(batch_size=batch_size))
    result = mcts.search(chess.Board(), simulations=137)
    assert mcts.root is not None
    assert result.playouts == result.nodes == mcts.root.visits == 137
    assert sum(result.visits.values()) == 136  # the root's own evaluation was the first
    assert max(evaluator.batch_sizes) <= batch_size
    assert all(node.virtual == 0 and not node.pending for node in walk(mcts.root))


def test_virtual_loss_fills_batches_with_different_positions() -> None:
    evaluator = UniformEvaluator()
    mcts = search(chess.Board(), evaluator, 400, batch_size=16)
    assert evaluator.batch_sizes[0] == 1  # the root, alone
    assert max(evaluator.batch_sizes) == 16
    assert mcts.root is not None
    expanded = sum(node.children is not None for node in walk(mcts.root))
    assert expanded == evaluator.positions  # no position was evaluated twice


def test_movetime_bounds_the_search() -> None:
    result = MCTS(UniformEvaluator()).search(chess.Board(), movetime=0.2)
    assert 0.2 <= result.seconds < 2.0
    assert result.playouts > 1
    assert result.nodes_per_second > 0


def test_a_budget_is_required_and_settings_are_checked() -> None:
    mcts = MCTS(UniformEvaluator())
    with pytest.raises(ValueError, match="simulations, movetime or a stop event"):
        mcts.search(chess.Board())
    with pytest.raises(ValueError, match="positive"):
        mcts.search(chess.Board(), simulations=0)
    for settings in ({"c_puct": 0.0}, {"fpu_reduction": -0.1}, {"batch_size": 0}):
        with pytest.raises(ValueError, match="invalid search settings"):
            SearchConfig(**settings)  # type: ignore[arg-type]


def test_the_principal_variation_is_the_most_visited_legal_line() -> None:
    board = chess.Board(MATE_IN_ONE[0])
    mcts = MCTS(UniformEvaluator())
    result = mcts.search(board, simulations=300)
    assert result.pv[0] == result.move
    replay = board.copy()
    for move in result.pv:
        assert move in replay.legal_moves
        replay.push(move)


# --- tree reuse ------------------------------------------------------------------------


def test_tree_reuse_keeps_the_visits_of_the_chosen_subtree() -> None:
    board = chess.Board()
    mcts = MCTS(MaterialEvaluator())
    result = mcts.search(board, simulations=600)
    assert mcts.root is not None
    assert mcts.root.children is not None
    played = mcts.root.children[result.move]
    reply = most_visited(played)
    assert played.children is not None
    kept = played.children[reply]
    visits = kept.visits
    assert visits > 1
    board.push(result.move)
    board.push(reply)
    again = mcts.search(board, simulations=100)
    assert mcts.root is kept
    assert again.nodes == mcts.root.visits == visits + 100
    assert again.playouts == 100


def test_a_board_from_another_game_starts_a_new_tree() -> None:
    mcts = MCTS(UniformEvaluator())
    e4 = chess.Board()
    e4.push_san("e4")
    mcts.search(e4, simulations=50)
    d4 = chess.Board()
    d4.push_san("d4")
    assert mcts.search(d4, simulations=50).nodes == 50
    same_position_no_history = chess.Board(e4.fen())
    mcts.search(e4, simulations=50)
    assert mcts.search(same_position_no_history, simulations=50).nodes == 50


# --- move choice -----------------------------------------------------------------------


def test_sample_move_follows_the_visit_counts() -> None:
    e4, d4, nf3 = (chess.Move.from_uci(uci) for uci in ("e2e4", "d2d4", "g1f3"))
    visits = {e4: 90, d4: 10, nf3: 0}
    rng = random.Random(0)
    assert sample_move(visits, 0.0, rng) == e4
    assert sample_move(visits, 1e-3, rng) == e4  # a tiny temperature does not overflow
    draws = Counter(sample_move(visits, 1.0, rng) for _ in range(2000))
    assert nf3 not in draws
    assert 0.85 < draws[e4] / 2000 < 0.95


def test_a_stop_event_ends_the_search_and_still_yields_a_move() -> None:
    stop = threading.Event()
    stop.set()  # already set: the first playout still runs
    result = MCTS(UniformEvaluator()).search(chess.Board(), stop=stop)
    assert result.playouts == 1
    assert result.move in chess.Board().legal_moves

    later = threading.Event()
    threading.Timer(0.2, later.set).start()
    started = time.monotonic()
    result = MCTS(UniformEvaluator()).search(chess.Board(), stop=later)
    assert 0.2 <= time.monotonic() - started < 2.0
    assert result.playouts > 1


def test_max_nodes_bounds_the_tree() -> None:
    mcts = MCTS(UniformEvaluator(), SearchConfig(batch_size=8))
    result = mcts.search(chess.Board(), simulations=10_000, max_nodes=300)
    assert 300 <= result.nodes < 300 + 8


class SlowEvaluator(UniformEvaluator):
    """Takes 20 ms a batch, so a search lasts a predictable time."""

    def evaluate(self, boards: Sequence[chess.Board]) -> list[Evaluation]:
        time.sleep(0.02)
        return super().evaluate(boards)


def test_progress_reports_arrive_during_a_search() -> None:
    reports: list[int] = []
    MCTS(SlowEvaluator(), SearchConfig(batch_size=4)).search(
        chess.Board(),
        simulations=200,  # 50 batches of 20 ms: about 1 s, so about 9 reports
        on_progress=lambda interim: reports.append(interim.playouts),
        progress_interval=0.1,
    )
    assert len(reports) >= 3
    assert reports == sorted(reports)
    assert all(0 < playouts < 200 for playouts in reports)


def test_the_result_reports_the_chosen_moves_value_and_a_mate_in_one() -> None:
    board = chess.Board(MATE_IN_ONE[0])
    result = MCTS(UniformEvaluator()).search(board, simulations=200)
    assert result.mate_in_one
    assert result.move_value == pytest.approx(1.0)  # the mated side's -1, for the mover
    quiet = MCTS(UniformEvaluator()).search(chess.Board(), simulations=50)
    assert not quiet.mate_in_one
