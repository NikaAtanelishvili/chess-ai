import logging
from pathlib import Path

import chess
import pytest
import torch
from torch import Tensor, nn

from chessnet.config import MODEL_PRESETS, ModelConfig, load_model_config
from chessnet.encoding.board import square_to_rank_file
from chessnet.encoding.moves import NUM_MOVE_TYPES, POLICY_SIZE, index_to_move, move_to_index
from chessnet.model.resnet import (
    ChessNet,
    PolicyHead,
    SqueezeExcitation,
    count_parameters,
    flatten_policy,
    scalar_value,
)

from helpers import positions

REPO = Path(__file__).parents[2]
log = logging.getLogger(__name__)

# Pinned so that any architecture change is deliberate (and documented in ARCHITECTURE.md).
PARAMETER_COUNTS = {"tiny": 626_380, "small": 2_294_092, "base": 3_526_092}

# Positions whose legal moves cover every move type: both colours, castling, en passant,
# promotions and underpromotions with and without capture, and 218 moves at once.
FLATTEN_FIXTURES = (
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


@pytest.mark.parametrize("name", MODEL_PRESETS)
def test_yaml_presets_match_the_code(name: str) -> None:
    assert load_model_config(REPO / "configs" / "model" / f"{name}.yaml") == MODEL_PRESETS[name]


@pytest.mark.parametrize("name", MODEL_PRESETS)
@pytest.mark.parametrize("batch", [1, 7])
def test_output_shapes(name: str, batch: int) -> None:
    network = ChessNet(MODEL_PRESETS[name])
    output = network(torch.randn(batch, 19, 8, 8))
    assert output.policy.shape == (batch, POLICY_SIZE)
    assert output.value.shape == (batch, 3)
    assert output.policy.dtype == output.value.dtype == torch.float32
    assert torch.isfinite(output.policy).all()
    assert torch.isfinite(output.value).all()


@pytest.mark.parametrize("name", MODEL_PRESETS)
def test_parameter_counts(name: str) -> None:
    count = count_parameters(ChessNet(MODEL_PRESETS[name]))
    log.info("%s: %s parameters", name, f"{count:,}")
    print(f"{name}: {count:,} parameters")  # shown with pytest -s
    assert count == PARAMETER_COUNTS[name]


def hot_logits(index: int) -> Tensor:
    """Head-shaped logits (1, 73, 8, 8) with one hot cell where ``index`` should land."""
    plane, from_square = divmod(index, 64)
    rank, file = square_to_rank_file(from_square)  # the encoding's orientation helper
    logits = torch.zeros(1, NUM_MOVE_TYPES, 8, 8)
    logits[0, plane, rank, file] = 1.0
    return logits


def legal_moves_of_fixtures() -> list[tuple[chess.Board, chess.Move]]:
    return [
        (board, move)
        for fen in FLATTEN_FIXTURES
        for board in [chess.Board(fen)]
        for move in board.legal_moves
    ]


def test_policy_flatten_order_matches_the_move_index() -> None:
    examples = legal_moves_of_fixtures()
    assert len(examples) > 400
    for board, move in examples:
        index = move_to_index(move, board)
        flat = flatten_policy(hot_logits(index))
        assert flat.shape == (1, POLICY_SIZE)
        assert int(flat.argmax()) == index
        assert index_to_move(int(flat.argmax()), board) == move


def test_policy_head_output_uses_the_move_index_order() -> None:
    head = PolicyHead(channels=8).eval()
    features = torch.randn(1, 8, 8, 8)
    for board, move in legal_moves_of_fixtures()[::7]:
        index = move_to_index(move, board)
        replacement = hot_logits(index)
        handle = head.out.register_forward_hook(
            lambda module, inputs, output, hot=replacement: hot  # replace the head's output
        )
        try:
            logits = head(features)
        finally:
            handle.remove()
        assert int(logits.argmax()) == index
        assert index_to_move(int(logits.argmax()), board) == move


def test_squeeze_excitation_can_be_disabled() -> None:
    with_se = ChessNet(ModelConfig(blocks=2, filters=16))
    without_se = ChessNet(ModelConfig(blocks=2, filters=16, se_ratio=None))
    assert any(isinstance(module, SqueezeExcitation) for module in with_se.modules())
    assert not any(isinstance(module, SqueezeExcitation) for module in without_se.modules())
    assert count_parameters(without_se) < count_parameters(with_se)
    assert without_se(torch.randn(2, 19, 8, 8)).policy.shape == (2, POLICY_SIZE)


def test_squeeze_excitation_scales_and_shifts_channels() -> None:
    se = SqueezeExcitation(channels=4, ratio=2)
    nn.init.zeros_(se.excite.weight)
    with torch.no_grad():
        se.excite.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]))
    x = torch.ones(1, 4, 8, 8)
    # sigmoid(0) = 0.5 scale, bias 1: every value becomes 0.5 * 1 + 1.
    assert torch.allclose(se(x), torch.full_like(x, 1.5))


@pytest.mark.parametrize(
    ("logits", "expected"),
    [
        ([20.0, 0.0, 0.0], 1.0),  # certain win
        ([0.0, 0.0, 20.0], -1.0),  # certain loss
        ([0.0, 20.0, 0.0], 0.0),  # certain draw
        ([1.0, 5.0, 1.0], 0.0),  # symmetric
    ],
)
def test_scalar_value(logits: list[float], expected: float) -> None:
    assert float(scalar_value(torch.tensor(logits))) == pytest.approx(expected, abs=1e-6)


def test_scalar_value_keeps_the_batch_shape() -> None:
    values = scalar_value(torch.randn(5, 3))
    assert values.shape == (5,)
    assert bool(((values >= -1) & (values <= 1)).all())


def test_same_seed_gives_the_same_network() -> None:
    torch.manual_seed(3)
    first = ChessNet(MODEL_PRESETS["tiny"])
    torch.manual_seed(3)
    second = ChessNet(MODEL_PRESETS["tiny"])
    for (name, a), (_, b) in zip(
        first.state_dict().items(), second.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), name
