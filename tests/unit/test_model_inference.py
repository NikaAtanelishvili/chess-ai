from pathlib import Path

import numpy as np
import pytest
import torch

from chessnet.config import MODEL_PRESETS
from chessnet.model.inference import InferenceModel, export_onnx
from chessnet.model.resnet import ChessNet


def test_inference_model_evaluates_without_autograd() -> None:
    network = ChessNet(MODEL_PRESETS["tiny"])
    assert network.training
    model = InferenceModel(network)
    assert not network.training  # switched to eval mode (BatchNorm uses running stats)
    planes = torch.randn(5, 19, 8, 8, requires_grad=True)
    output = model(planes)
    assert output.policy.shape == (5, 4672)
    assert not output.policy.requires_grad
    assert torch.is_inference(output.policy)
    with torch.no_grad():
        expected = network(planes)
    assert torch.equal(output.policy, expected.policy)
    assert torch.equal(output.value, expected.value)


def test_onnx_opset_must_be_recent(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 17"):
        export_onnx(ChessNet(MODEL_PRESETS["tiny"]), tmp_path / "model.onnx", opset=16)


@pytest.mark.slow
def test_onnx_export_matches_pytorch(tmp_path: Path) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    network = ChessNet(MODEL_PRESETS["tiny"])
    path = export_onnx(network, tmp_path / "tiny.onnx")
    assert network.training  # training mode is restored after the export
    assert [p.name for p in tmp_path.iterdir()] == ["tiny.onnx"]  # weights inside the file

    session = onnxruntime.InferenceSession(str(path))
    planes = torch.randn(3, 19, 8, 8)  # a batch size other than the export example's
    policy, value = session.run(["policy", "value"], {"planes": planes.numpy()})
    with torch.no_grad():
        expected = network.eval()(planes)
    np.testing.assert_allclose(policy, expected.policy.numpy(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(value, expected.value.numpy(), atol=1e-4, rtol=1e-4)


@pytest.mark.slow
def test_compiled_inference_matches_eager() -> None:
    network = ChessNet(MODEL_PRESETS["tiny"])
    planes = torch.randn(4, 19, 8, 8)
    eager = InferenceModel(network)(planes)
    compiled = InferenceModel(network, compile=True)(planes)
    torch.testing.assert_close(compiled.policy, eager.policy, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(compiled.value, eager.value, atol=1e-4, rtol=1e-4)
