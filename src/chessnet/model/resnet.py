"""Policy/value network: an SE-ResNet trunk with a convolutional policy head and a WDL value head.

* Trunk (Leela Chess Zero style): a 3x3 input convolution, then residual blocks of
  conv-BN-ReLU-conv-BN, squeeze-excitation, skip connection and ReLU.
* Policy head: 3x3 convolution (F -> F), BN, ReLU, 1x1 convolution (F -> 73), flattened to
  4672 logits in the order of :mod:`chessnet.encoding.moves` (``plane * 64 + square``).
* Value head: 1x1 convolution (F -> 32), BN, ReLU, fully connected 2048 -> 128, ReLU, and
  128 -> 3 logits for win, draw and loss from the side to move's perspective.

Inputs are the ``(N, 19, 8, 8)`` planes of :mod:`chessnet.encoding.board`.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F  # noqa: N812 (conventional alias)
from torch import Tensor, nn

from chessnet.config import ModelConfig
from chessnet.encoding.board import NUM_PLANES
from chessnet.encoding.moves import NUM_MOVE_TYPES
from chessnet.encoding.value import LOSS, WDL_CLASSES, WIN


class PolicyValue(NamedTuple):
    """Network outputs: ``policy`` logits ``(N, 4672)`` and ``value`` WDL logits ``(N, 3)``."""

    policy: Tensor
    value: Tensor


def flatten_policy(logits: Tensor) -> Tensor:
    """Flatten policy-head logits ``(N, 73, 8, 8)`` to ``(N, 4672)``.

    Row-major flattening over ``[plane, rank, file]`` gives ``plane * 64 + 8 * rank + file``,
    which is ``plane * 64 + square``: exactly the index of :func:`~chessnet.encoding.moves.
    move_to_index`.
    """
    return logits.flatten(start_dim=1)


def scalar_value(wdl_logits: Tensor) -> Tensor:
    """Expected outcome ``P(win) - P(loss)`` in ``[-1, 1]`` from WDL logits ``(..., 3)``."""
    probabilities = torch.softmax(wdl_logits, dim=-1)
    return probabilities[..., WIN] - probabilities[..., LOSS]


class SqueezeExcitation(nn.Module):
    """Channel-wise gating (Leela style): a learned per-channel scale and bias.

    Global average pooling, then ``C -> C/ratio`` (ReLU) and ``C/ratio -> 2C``; one half
    becomes a sigmoid scale, the other a bias.
    """

    def __init__(self, channels: int, ratio: int) -> None:
        super().__init__()
        self.squeeze = nn.Linear(channels, max(channels // ratio, 1))
        self.excite = nn.Linear(max(channels // ratio, 1), 2 * channels)

    def forward(self, x: Tensor) -> Tensor:
        pooled = x.mean(dim=(2, 3))
        scale, bias = self.excite(F.relu(self.squeeze(pooled))).chunk(2, dim=1)
        gated: Tensor = x * torch.sigmoid(scale)[:, :, None, None] + bias[:, :, None, None]
        return gated


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, se_ratio: int | None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se: nn.Module = SqueezeExcitation(channels, se_ratio) if se_ratio else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        residual: Tensor = self.se(self.bn2(self.conv2(out)))
        return F.relu(residual + x)


class PolicyHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.out = nn.Conv2d(channels, NUM_MOVE_TYPES, 1)

    def forward(self, x: Tensor) -> Tensor:
        logits: Tensor = self.out(F.relu(self.bn(self.conv(x))))
        return flatten_policy(logits)


class ValueHead(nn.Module):
    def __init__(self, channels: int, value_channels: int, hidden: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, value_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(value_channels)
        self.fc1 = nn.Linear(value_channels * 64, hidden)
        self.fc2 = nn.Linear(hidden, len(WDL_CLASSES))

    def forward(self, x: Tensor) -> Tensor:
        features = F.relu(self.bn(self.conv(x))).flatten(start_dim=1)
        logits: Tensor = self.fc2(F.relu(self.fc1(features)))
        return logits


class ChessNet(nn.Module):
    """The policy/value network, built from a :class:`~chessnet.config.ModelConfig`."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, config.filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(config.filters),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            *(ResidualBlock(config.filters, config.se_ratio) for _ in range(config.blocks))
        )
        self.policy_head = PolicyHead(config.filters)
        self.value_head = ValueHead(config.filters, config.value_channels, config.value_hidden)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, planes: Tensor) -> PolicyValue:
        features: Tensor = self.trunk(self.stem(planes))
        policy: Tensor = self.policy_head(features)
        value: Tensor = self.value_head(features)
        return PolicyValue(policy, value)


def count_parameters(module: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
