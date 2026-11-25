"""Evaluation metrics, accumulated over batches.

The White/Black split of top-1 accuracy is a permanent guard against the original defect:
with a correct side-to-move encoding, both colours are predicted equally well, so a gap of
more than about a point means something is wrong with mirroring.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch.nn.functional as F  # noqa: N812 (conventional alias)
from torch import Tensor

from chessnet.model.resnet import PolicyValue, scalar_value

#: Game-phase buckets for accuracy by ply: [start, end) in plies.
PHASES = {"opening": (0, 20), "middlegame": (20, 61), "endgame": (61, 1 << 16)}


def topk_correct(
    logits: Tensor, targets: Tensor, ks: Sequence[int] = (1, 3, 5)
) -> dict[int, Tensor]:
    """Per-sample booleans: is the target among the ``k`` highest logits?"""
    ranked = logits.topk(min(max(ks), logits.shape[1]), dim=1).indices  # k <= classes
    hits = ranked == targets.unsqueeze(1)
    return {k: hits[:, :k].any(dim=1) for k in ks}


def value_targets(wdl: Tensor) -> Tensor:
    """Scalar outcome for the mover: win 1, draw 0, loss -1 (WDL classes 0, 1, 2)."""
    return 1.0 - wdl.float()


class MetricAccumulator:
    """Sums metrics over validation batches; :meth:`compute` returns the averages."""

    def __init__(self) -> None:
        self.count = 0
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def _add(self, name: str, total: float, count: int) -> None:
        self.sums[name] = self.sums.get(name, 0.0) + total
        self.counts[name] = self.counts.get(name, 0) + count

    def update(
        self,
        output: PolicyValue,
        moves: Tensor,
        wdl: Tensor,
        ply: Tensor,
        *,
        label_smoothing: float = 0.0,
        value_weight: float = 1.0,
    ) -> None:
        policy_logits = output.policy.float()
        value_logits = output.value.float()
        n = int(moves.shape[0])
        self.count += n
        policy_loss = F.cross_entropy(
            policy_logits, moves, label_smoothing=label_smoothing, reduction="sum"
        )
        value_loss = F.cross_entropy(value_logits, wdl, reduction="sum")
        self._add("policy_loss", float(policy_loss), n)
        self._add("value_loss", float(value_loss), n)
        self._add("loss", float(policy_loss + value_weight * value_loss), n)

        hits = topk_correct(policy_logits, moves)
        for k, correct in hits.items():
            self._add(f"top{k}", float(correct.sum()), n)
        self._add("wdl_accuracy", float((value_logits.argmax(dim=1) == wdl).sum()), n)
        error = scalar_value(value_logits) - value_targets(wdl)
        self._add("value_mse", float((error * error).sum()), n)

        top1 = hits[1]
        white = ply % 2 == 0  # ply 0 is White's first move
        for name, mask in (("top1_white", white), ("top1_black", ~white)):
            self._add(name, float(top1[mask].sum()), int(mask.sum()))
        for name, (start, end) in PHASES.items():
            mask = (ply >= start) & (ply < end)
            self._add(f"top1_{name}", float(top1[mask].sum()), int(mask.sum()))

    def compute(self) -> dict[str, float]:
        """Averages; a bucket without samples is reported as NaN."""
        metrics = {
            name: total / self.counts[name] if self.counts[name] else float("nan")
            for name, total in self.sums.items()
        }
        if "top1_white" in metrics:
            metrics["top1_colour_gap"] = abs(metrics["top1_white"] - metrics["top1_black"])
        return metrics
