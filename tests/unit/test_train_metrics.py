import math

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from chessnet.model.resnet import PolicyValue
from chessnet.train.losses import compute_losses
from chessnet.train.metrics import MetricAccumulator, topk_correct, value_targets


def test_topk_on_hand_made_logits() -> None:
    logits = torch.tensor(
        [
            [0.1, 0.9, 0.3, 0.2, 0.0, 0.5],  # ranking: 1, 5, 2, 3, 0, 4
            [0.6, 0.1, 0.2, 0.3, 0.4, 0.5],  # ranking: 0, 5, 4, 3, 2, 1
        ]
    )
    targets = torch.tensor([2, 1])
    hits = topk_correct(logits, targets, ks=(1, 3, 5))
    assert hits[1].tolist() == [False, False]
    assert hits[3].tolist() == [True, False]  # target 2 is third in row 0
    assert hits[5].tolist() == [True, False]  # target 1 is last in row 1
    assert topk_correct(logits, torch.tensor([1, 0]), ks=(1,))[1].tolist() == [True, True]


def test_value_targets() -> None:
    assert value_targets(torch.tensor([0, 1, 2])).tolist() == [1.0, 0.0, -1.0]


def output(policy: list[list[float]], value: list[list[float]]) -> PolicyValue:
    return PolicyValue(torch.tensor(policy), torch.tensor(value))


def test_accumulator_splits_by_colour_and_phase() -> None:
    accumulator = MetricAccumulator()
    # Four samples: plies 0 (White, opening), 1 (Black, opening), 40 (White, middlegame),
    # 81 (Black, endgame). Policy correct for samples 0 and 2 (both White to move).
    accumulator.update(
        output(
            [[5.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 0.0]],
            [[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0], [9.0, 0.0, 0.0]],
        ),
        moves=torch.tensor([0, 1, 1, 1]),
        wdl=torch.tensor([0, 1, 2, 2]),
        ply=torch.tensor([0, 1, 40, 81]),
    )
    metrics = accumulator.compute()
    assert metrics["top1"] == pytest.approx(0.5)
    assert metrics["top1_white"] == pytest.approx(1.0)
    assert metrics["top1_black"] == pytest.approx(0.0)
    assert metrics["top1_colour_gap"] == pytest.approx(1.0)
    assert metrics["top1_opening"] == pytest.approx(0.5)
    assert metrics["top1_middlegame"] == pytest.approx(1.0)
    assert metrics["top1_endgame"] == pytest.approx(0.0)
    assert metrics["wdl_accuracy"] == pytest.approx(0.75)  # the last sample predicts a win
    assert accumulator.count == 4


def test_accumulator_averages_over_batches() -> None:
    accumulator = MetricAccumulator()
    for move in (0, 1):
        accumulator.update(
            output([[3.0, 0.0]], [[0.0, 0.0, 0.0]]),
            moves=torch.tensor([move]),
            wdl=torch.tensor([1]),
            ply=torch.tensor([10]),
        )
    metrics = accumulator.compute()
    assert metrics["top1"] == pytest.approx(0.5)
    expected_policy = (
        F.cross_entropy(torch.tensor([[3.0, 0.0]]), torch.tensor([0]))
        + F.cross_entropy(torch.tensor([[3.0, 0.0]]), torch.tensor([1]))
    ) / 2
    assert metrics["policy_loss"] == pytest.approx(float(expected_policy))
    assert metrics["value_mse"] == pytest.approx(0.0)  # uniform WDL: v = 0 = draw target
    assert math.isnan(metrics["top1_black"])  # no Black-to-move sample
    assert math.isnan(metrics["top1_colour_gap"])


def test_losses_weight_the_value_term() -> None:
    out = output([[2.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    moves, wdl = torch.tensor([0]), torch.tensor([1])
    plain = compute_losses(out, moves, wdl, value_weight=1.0, label_smoothing=0.0)
    half = compute_losses(out, moves, wdl, value_weight=0.5, label_smoothing=0.0)
    assert float(half.total) == pytest.approx(float(plain.policy + 0.5 * plain.value))
    assert float(plain.total) == pytest.approx(float(plain.policy + plain.value))
    smoothed = compute_losses(out, moves, wdl, value_weight=1.0, label_smoothing=0.1)
    assert float(smoothed.policy) > float(plain.policy)
    assert float(smoothed.value) == pytest.approx(float(plain.value))
