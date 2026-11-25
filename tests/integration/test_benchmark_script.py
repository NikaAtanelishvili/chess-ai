"""The training benchmark script still runs (it is used in the first Kaggle GPU session)."""

import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from helpers.datasets import make_dataset

SCRIPT = Path(__file__).parents[2] / "scripts" / "benchmark_train.py"
QUICK = ["--models", "tiny", "--steps", "2", "--warmup", "1", "--device", "cpu"]


def test_benchmark_script_writes_a_report(tmp_path: Path) -> None:
    data = make_dataset(tmp_path / "shards", positions=600, shard_size=300)
    report = tmp_path / "benchmark.md"
    command = [
        sys.executable,
        str(SCRIPT),
        *QUICK,
        "--batch-sizes",
        "16",
        "32",
        "--compile",
        "off",  # compiling on the CPU takes a while; the slow training test covers it
        "--data",
        str(data),
        "--loader-batches",
        "2",
        "--workers",
        "0",
        "--out",
        str(report),
        "--json",
        str(tmp_path / "benchmark.json"),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
    text = report.read_text(encoding="utf-8")
    for batch, channels_last in itertools.product((16, 32), ("off", "on")):
        assert f"| tiny | 626,380 | {batch} | {channels_last} | off |" in text
    assert "- tiny: batch " in text  # the fastest configuration
    assert "- 0 workers, batch 32: " in text
    measured = json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8"))
    assert measured["failures"] == []
    assert len(measured["results"]) == 4
    for result in measured["results"]:
        assert result["train_speed"] > 0
        assert result["eval_speed"] > 0  # every row times validation's forward passes
        assert result["memory"] is None  # no GPU


def load_script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark_train", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)  # dataclasses look the module up
    spec.loader.exec_module(module)
    return module


def test_a_compile_failure_is_reported_and_the_benchmark_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = load_script(monkeypatch)
    prepare_network = benchmark.prepare_network

    def without_compiler(network: torch.nn.Module, *, channels_last: bool, compile: bool) -> object:
        if compile:
            raise RuntimeError("no compiler")  # stands in for an Inductor or Triton failure
        return prepare_network(network, channels_last=channels_last, compile=compile)

    monkeypatch.setattr(benchmark, "prepare_network", without_compiler)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", torch.backends.cudnn.benchmark)
    report = tmp_path / "benchmark.md"
    arguments = [*QUICK, "--batch-sizes", "16", "--channels-last", "off", "--out", str(report)]
    arguments += ["--json", str(tmp_path / "benchmark.json")]
    monkeypatch.setattr(sys, "argv", ["benchmark_train.py", *arguments])
    benchmark.main()
    text = report.read_text(encoding="utf-8")
    assert "| tiny | 626,380 | 16 | off | on | compile failed: RuntimeError |" in text
    assert "- tiny: batch 16, channels_last off, compile off: " in text
    measured = json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8"))
    assert [result["compile"] for result in measured["results"]] == [False]
    [failure] = measured["failures"]
    assert failure["error"] == "compile failed: RuntimeError"
    assert failure["compile"] is True
