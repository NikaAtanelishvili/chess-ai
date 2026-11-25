"""Training loss: policy cross-entropy plus weighted value (WDL) cross-entropy."""

from __future__ import annotations

from typing import NamedTuple

import torch.nn.functional as F  # noqa: N812 (conventional alias)
from torch import Tensor

from chessnet.model.resnet import PolicyValue


class Losses(NamedTuple):
    total: Tensor
    policy: Tensor
    value: Tensor


def compute_losses(
    output: PolicyValue,
    moves: Tensor,
    wdl: Tensor,
    *,
    value_weight: float,
    label_smoothing: float,
) -> Losses:
    """``CE(policy, move, label_smoothing) + value_weight * CE(value, wdl)``.

    No legal-move mask is applied: the network learns legality quickly, and masking would
    need python-chess per sample. Logits are cast to float32 so the losses stay accurate
    under fp16 autocast.
    """
    policy = F.cross_entropy(output.policy.float(), moves, label_smoothing=label_smoothing)
    value = F.cross_entropy(output.value.float(), wdl)
    return Losses(policy + value_weight * value, policy, value)
