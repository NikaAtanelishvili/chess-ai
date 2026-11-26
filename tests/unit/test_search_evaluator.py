"""The network evaluator: priors over exactly the legal moves, values in range, the cache."""

import copy
from pathlib import Path

import chess
import pytest
import torch

from chessnet.config import MODEL_PRESETS
from chessnet.encoding.moves import mirror_move
from chessnet.model.checkpoint import save_checkpoint
from chessnet.model.resnet import ChessNet
from chessnet.search.evaluator import NetworkEvaluator, position_key

from helpers import positions

FENS = (
    positions.START,
    positions.CASTLING_WHITE,
    positions.CASTLING_BLACK,
    positions.EN_PASSANT_WHITE,
    positions.EN_PASSANT_BLACK,
    positions.PROMOTION_WHITE,
    positions.PROMOTION_BLACK,
    positions.PROMOTION_RICH,
    positions.MAX_MOBILITY,
)


@pytest.fixture(scope="module")
def network() -> ChessNet:
    torch.manual_seed(0)
    return ChessNet(MODEL_PRESETS["tiny"])


@pytest.fixture
def evaluator(network: ChessNet) -> NetworkEvaluator:
    return NetworkEvaluator(network)


@pytest.mark.parametrize("fen", FENS)
def test_priors_cover_exactly_the_legal_moves_and_sum_to_one(
    evaluator: NetworkEvaluator, fen: str
) -> None:
    board = chess.Board(fen)
    [evaluation] = evaluator.evaluate([board])
    assert set(evaluation.priors) == set(board.legal_moves)  # no illegal keys, none missing
    assert sum(evaluation.priors.values()) == pytest.approx(1.0)
    assert all(prior > 0 for prior in evaluation.priors.values())
    assert -1.0 <= evaluation.value <= 1.0


def test_a_finished_game_has_no_priors(evaluator: NetworkEvaluator) -> None:
    [evaluation] = evaluator.evaluate([chess.Board("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1")])
    assert evaluation.priors == {}


def test_a_batch_matches_single_evaluations(network: ChessNet) -> None:
    boards = [chess.Board(fen) for fen in FENS]
    batched = NetworkEvaluator(network).evaluate(boards)
    for board, together in zip(boards, batched, strict=True):
        [alone] = NetworkEvaluator(network).evaluate([board])
        assert together.value == pytest.approx(alone.value, abs=1e-5)
        for move, prior in alone.priors.items():
            assert together.priors[move] == pytest.approx(prior, abs=1e-5)


def test_mirrored_positions_get_mirrored_priors_and_the_same_value(
    evaluator: NetworkEvaluator,
) -> None:
    for fen in FENS:
        board = chess.Board(fen)
        [original, mirrored] = evaluator.evaluate([board, board.mirror()])
        assert mirrored.value == pytest.approx(original.value, abs=1e-6)
        for move, prior in original.priors.items():
            assert mirrored.priors[mirror_move(move)] == pytest.approx(prior, abs=1e-6)


def test_the_cache_answers_repeated_positions(network: ChessNet) -> None:
    evaluator = NetworkEvaluator(network)
    boards = [chess.Board(fen) for fen in FENS[:3]]
    first = evaluator.evaluate(boards)
    assert (evaluator.positions, evaluator.batches) == (3, 1)
    second = evaluator.evaluate([*boards, boards[0]])
    assert (evaluator.positions, evaluator.batches) == (3, 1)  # nothing new to evaluate
    assert second[:3] == first
    assert second[3] == first[0]


def test_duplicates_within_a_batch_are_evaluated_once(network: ChessNet) -> None:
    evaluator = NetworkEvaluator(network)
    transposed = chess.Board()
    for san in ("Nf3", "Nf6", "Nc3", "Nc6"):
        transposed.push_san(san)
    other_order = chess.Board()
    for san in ("Nc3", "Nc6", "Nf3", "Nf6"):
        other_order.push_san(san)
    results = evaluator.evaluate([transposed, other_order])
    assert evaluator.positions == 1
    assert results[0] == results[1]


def test_the_cache_is_bounded(network: ChessNet) -> None:
    evaluator = NetworkEvaluator(network, cache_size=2)
    boards = [chess.Board(fen) for fen in FENS[:3]]
    for board in boards:
        evaluator.evaluate([board])
    evaluator.evaluate([boards[0]])  # evicted as the oldest entry
    assert evaluator.positions == 4
    evaluator.evaluate([boards[2]])  # still cached
    assert evaluator.positions == 4
    uncached = NetworkEvaluator(network, cache_size=0)
    uncached.evaluate([boards[0]])
    uncached.evaluate([boards[0]])
    assert uncached.positions == 2


def test_position_key_holds_exactly_what_the_network_sees() -> None:
    a, b = chess.Board(), chess.Board()
    for san in ("Nf3", "Nf6", "Ng1", "Ng8"):
        b.push_san(san)
    assert position_key(a) != position_key(b)  # same pieces, but a halfmove clock of 4
    assert position_key(chess.Board(a.fen())) == position_key(a)  # history is not part of it
    clock = "4k3/8/8/8/8/8/8/R3K3 w - - {} 80"
    assert position_key(chess.Board(clock.format(120))) == position_key(
        chess.Board(clock.format(100))
    )  # the clock feature saturates at 100
    assert position_key(chess.Board(clock.format(99))) != position_key(
        chess.Board(clock.format(100))
    )
    # An en-passant square without a legal capture is not seen either.
    no_capture = "4k3/8/8/8/4P3/8/8/4K3 b - e3 0 1"
    assert position_key(chess.Board(no_capture)) == position_key(
        chess.Board(no_capture.replace("e3", "-"))
    )
    with_capture = positions.EN_PASSANT_WHITE
    assert position_key(chess.Board(with_capture)) != position_key(
        chess.Board(with_capture.replace(" f6 ", " - "))
    )


def test_from_checkpoint(network: ChessNet, tmp_path: Path) -> None:
    path = save_checkpoint(tmp_path / "tiny.pt", network)
    evaluator = NetworkEvaluator.from_checkpoint(path, cache_size=10)
    [evaluation] = evaluator.evaluate([chess.Board()])
    [expected] = NetworkEvaluator(network).evaluate([chess.Board()])
    assert evaluation.value == pytest.approx(expected.value, abs=1e-6)
    assert evaluator.cache_size == 10


@pytest.mark.gpu
def test_half_precision_on_cuda_matches_the_cpu(network: ChessNet) -> None:
    boards = [chess.Board(fen) for fen in FENS]
    on_cpu = NetworkEvaluator(copy.deepcopy(network)).evaluate(boards)
    cuda = NetworkEvaluator(copy.deepcopy(network), device="cuda")
    assert cuda.half
    for expected, actual in zip(on_cpu, cuda.evaluate(boards), strict=True):
        assert actual.value == pytest.approx(expected.value, abs=2e-2)
        assert set(actual.priors) == set(expected.priors)
        for move, prior in expected.priors.items():
            assert actual.priors[move] == pytest.approx(prior, abs=2e-2)
