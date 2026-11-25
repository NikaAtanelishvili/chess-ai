import copy
import itertools
import signal
import time
from pathlib import Path

import pytest
import torch

from chessnet.config import MODEL_PRESETS, TrainConfig
from chessnet.model.resnet import ChessNet
from chessnet.train.loop import (
    StopRequest,
    find_checkpoint,
    make_optimizer,
    model_input,
    parse_duration,
    prepare_network,
    warmup_cosine,
)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("11h15m", 40_500),
        ("90m", 5_400),
        ("45s", 45),
        ("1h", 3_600),
        ("1h0m30s", 3_630),
        ("0.5h", 1_800),
        ("3600", 3_600),
        (" 2H ", 7_200),
    ],
)
def test_parse_duration(text: str, seconds: float) -> None:
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "h", "10x", "1m2h", "-5m"])
def test_parse_duration_rejects_nonsense(text: str) -> None:
    with pytest.raises(ValueError, match="not a duration"):
        parse_duration(text)


def test_warmup_cosine_schedule() -> None:
    multipliers = [warmup_cosine(step, warmup=10, total=110, min_ratio=0.1) for step in range(120)]
    assert multipliers[0] == pytest.approx(0.1)
    assert multipliers[9] == pytest.approx(1.0)  # end of warmup
    assert multipliers[10] == pytest.approx(1.0)  # cosine starts at the peak
    assert multipliers[60] == pytest.approx(0.55)  # halfway: (1 + 0.1) / 2
    assert multipliers[110] == pytest.approx(0.1)  # floor reached at the last step
    assert multipliers[119] == pytest.approx(0.1)  # and kept afterwards
    decay = multipliers[10:111]
    assert all(a >= b for a, b in itertools.pairwise(decay))


def test_optimizer_skips_weight_decay_for_biases_and_norms() -> None:
    network = ChessNet(MODEL_PRESETS["tiny"])
    adamw = make_optimizer(TrainConfig(model="tiny", weight_decay=0.01), network)
    assert isinstance(adamw, torch.optim.AdamW)
    decayed, plain = adamw.param_groups
    assert decayed["weight_decay"] == 0.01
    assert plain["weight_decay"] == 0.0
    assert all(p.ndim > 1 for p in decayed["params"])
    assert all(p.ndim <= 1 for p in plain["params"])
    count = sum(p.numel() for group in adamw.param_groups for p in group["params"])
    assert count == sum(p.numel() for p in network.parameters())

    sgd = make_optimizer(TrainConfig(model="tiny", optimizer="sgd", momentum=0.8), network)
    assert isinstance(sgd, torch.optim.SGD)
    assert sgd.param_groups[0]["nesterov"]
    assert sgd.param_groups[0]["momentum"] == 0.8


def test_stop_request_reasons() -> None:
    started = time.monotonic()
    stop = StopRequest(max_wall_time=60.0, started=started)
    assert stop.reason(started + 59, step=10, stop_at_step=None) is None
    assert stop.reason(started + 60, step=10, stop_at_step=None) == "max_wall_time"
    assert stop.reason(started, step=10, stop_at_step=10) == "stop_at_step"
    unlimited = StopRequest(max_wall_time=None, started=started)
    assert unlimited.reason(started + 1e9, step=10, stop_at_step=None) is None


def test_sigterm_requests_a_stop_and_handlers_are_restored() -> None:
    before = signal.getsignal(signal.SIGTERM)
    stop = StopRequest(max_wall_time=None, started=time.monotonic())
    with stop.handling_signals():
        signal.raise_signal(signal.SIGTERM)
        assert stop.reason(time.monotonic(), step=1, stop_at_step=None) == "signal"
        assert stop.signal_name == "SIGTERM"
    assert signal.getsignal(signal.SIGTERM) == before


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_find_checkpoint(tmp_path: Path) -> None:
    out = tmp_path / "run"
    previous = tmp_path / "previous" / "run"
    assert find_checkpoint("auto", out, []) is None
    touch(out / "checkpoints" / "step-000000100.pt")
    newest = touch(previous / "checkpoints" / "step-000000300.pt")
    touch(out / "best.pt")  # never picked by "auto"
    assert find_checkpoint("auto", out, [tmp_path / "previous", tmp_path / "missing"]) == newest
    assert find_checkpoint("auto", out, []) == out / "checkpoints" / "step-000000100.pt"
    assert find_checkpoint("none", out, [tmp_path / "previous"]) is None
    assert find_checkpoint(str(out / "best.pt"), out, []) == out / "best.pt"
    with pytest.raises(FileNotFoundError):
        find_checkpoint(str(out / "nope.pt"), out, [])


def test_channels_last_keeps_the_outputs_and_the_policy_order() -> None:
    torch.manual_seed(0)
    reference = ChessNet(MODEL_PRESETS["tiny"]).eval()
    network = copy.deepcopy(reference)
    model = prepare_network(network, channels_last=True, compile=False)
    assert model is network
    kernels = [p for p in network.parameters() if p.ndim == 4]
    assert kernels
    assert all(p.is_contiguous(memory_format=torch.channels_last) for p in kernels)

    planes = (torch.rand(4, 19, 8, 8) < 0.3).float()
    assert model_input(planes, channels_last=False) is planes
    nhwc = model_input(planes, channels_last=True)
    assert nhwc.is_contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        expected, actual = reference(planes), model(nhwc)
    # A flatten that followed the memory layout would scramble the move index, not perturb it.
    torch.testing.assert_close(actual.policy, expected.policy, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(actual.value, expected.value, rtol=1e-4, atol=1e-5)


def test_compiling_leaves_weights_and_checkpoint_keys_in_the_network() -> None:
    network = ChessNet(MODEL_PRESETS["tiny"])
    keys = list(network.state_dict())
    model = prepare_network(network, channels_last=False, compile=True)  # compiles lazily
    assert model is not network
    assert list(network.state_dict()) == keys  # checkpoints save the network, not the wrapper
    assert all(a is b for a, b in zip(model.parameters(), network.parameters(), strict=True))
