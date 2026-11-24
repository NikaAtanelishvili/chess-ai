"""Running a trained network: an inference wrapper and ONNX export (for fast CPU play)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from chessnet.encoding.board import NUM_PLANES
from chessnet.model.resnet import ChessNet, PolicyValue

#: Lowest ONNX opset the export supports.
MIN_ONNX_OPSET = 17


class InferenceModel:
    """A network prepared for evaluation: eval mode, no autograd, optionally compiled.

    Calling it with input planes ``(N, 19, 8, 8)`` returns :class:`PolicyValue` logits.
    """

    def __init__(
        self, network: ChessNet, *, device: str | torch.device = "cpu", compile: bool = False
    ) -> None:
        self.device = torch.device(device)
        self.network = network.to(self.device).eval()
        self._forward: Callable[[Tensor], PolicyValue] = (
            cast(Callable[[Tensor], PolicyValue], torch.compile(self.network))
            if compile
            else self.network
        )

    def __call__(self, planes: Tensor) -> PolicyValue:
        with torch.inference_mode():
            return self._forward(planes.to(self.device, non_blocking=True))


def export_onnx(network: ChessNet, path: str | Path, *, opset: int = 18) -> Path:
    """Export ``network`` (in eval mode) to a single ONNX file with a dynamic batch size.

    Inputs: ``planes`` ``(batch, 19, 8, 8)`` float32. Outputs: ``policy`` ``(batch, 4672)``
    and ``value`` ``(batch, 3)`` logits. Needs the ``onnx`` extra (onnx and onnxscript).
    """
    if opset < MIN_ONNX_OPSET:
        raise ValueError(f"ONNX opset must be at least {MIN_ONNX_OPSET}, got {opset}")
    path = Path(path)
    was_training = network.training
    network.eval()
    try:
        example = torch.zeros(2, NUM_PLANES, 8, 8)
        torch.onnx.export(
            network,
            (example,),
            str(path),
            input_names=["planes"],
            output_names=["policy", "value"],
            opset_version=opset,
            dynamic_shapes={"planes": {0: torch.export.Dim("batch")}},
            external_data=False,
            dynamo=True,
        )
    finally:
        network.train(was_training)
    return path
