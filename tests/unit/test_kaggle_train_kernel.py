"""The Kaggle training kernel: input discovery, and the two runs it supervises.

The main run must never be stopped, slowed or abandoned because of the second run.
"""

import importlib.util
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from chessnet.config import load_train_config
from chessnet.model.checkpoint import load_checkpoint
from chessnet.train.loop import parse_duration

from helpers.datasets import make_dataset

REPO = Path(__file__).parents[2]
KERNEL = REPO / "kaggle" / "train" / "train.py"
FAST_SPEED = {"train_speed": 2_000.0, "eval_speed": 4_000.0, "first_step": 1.0}


@pytest.fixture
def kernel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("kaggle_train_kernel", KERNEL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # defines functions only; main() is guarded
    monkeypatch.setattr(module, "INPUT", tmp_path / "input")
    monkeypatch.setattr(module, "WORKING", tmp_path / "working")
    monkeypatch.setattr(module, "CHECKOUT", REPO)
    monkeypatch.setattr(module, "POLL", 0.05)
    (tmp_path / "input").mkdir()
    (tmp_path / "working").mkdir()
    return module


def add_file(root: Path, relative: str, text: str = "{}") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def python(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def start(code: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(python(code))


# --- inputs ----------------------------------------------------------------------------


def test_finds_the_single_manifest_wherever_it_is_mounted(
    kernel: ModuleType, tmp_path: Path
) -> None:
    manifest = add_file(tmp_path, "input/datasets/someone/chessnet-lichess/shards/manifest.json")
    add_file(tmp_path, "input/chessnet-run/run-base/runs.jsonl")  # a previous run: no manifest
    assert kernel.find_one("**/manifest.json", "manifests", required=True) == manifest.parent


def test_no_manifest_is_a_clear_error(kernel: ModuleType) -> None:
    with pytest.raises(SystemExit, match=r"No dataset manifests .*dataset_sources"):
        kernel.find_one("**/manifest.json", "dataset manifests", required=True)


def test_two_manifests_are_a_clear_error(kernel: ModuleType, tmp_path: Path) -> None:
    add_file(tmp_path, "input/a/shards/manifest.json")
    add_file(tmp_path, "input/b/shards/manifest.json")
    with pytest.raises(SystemExit, match="Found 2 manifests"):
        kernel.find_one("**/manifest.json", "manifests", required=True)


def test_previous_runs_are_restored_into_their_own_directories(
    kernel: ModuleType, tmp_path: Path
) -> None:
    kernel.restore_previous_runs()  # nothing attached: nothing to restore
    assert not list((tmp_path / "working").iterdir())
    add_file(tmp_path, "input/chessnet-run/run-base/runs.jsonl", "base")
    add_file(tmp_path, "input/chessnet-run/run-base/checkpoints/step-000000010.pt", "b10")
    add_file(tmp_path, "input/chessnet-run/run-wide/runs.jsonl", "wide")
    add_file(tmp_path, "input/chessnet-run/run-wide/config.yaml", "model: wide\n")
    kernel.restore_previous_runs()
    working = tmp_path / "working"
    assert (working / "run-base" / "checkpoints" / "step-000000010.pt").read_text() == "b10"
    assert (working / "run-wide" / "config.yaml").read_text() == "model: wide\n"
    assert not (working / "run-base" / "config.yaml").exists()


def test_an_unset_ref_stops_before_doing_anything(
    kernel: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kernel, "CHESSNET_REF", "<tag-or-commit-sha>")
    monkeypatch.setattr(kernel, "run", lambda *args, **kwargs: pytest.fail("must not run"))
    with pytest.raises(SystemExit, match="Set CHESSNET_REF"):
        kernel.main()


def test_an_unknown_mode_is_refused(kernel: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kernel, "CHESSNET_REF", "0" * 40)
    monkeypatch.setattr(kernel, "MODE", "full")
    with pytest.raises(SystemExit, match="MODE must be one of"):
        kernel.main()


def test_the_wall_time_constant_matches_the_main_runs_flag(kernel: ModuleType) -> None:
    assert parse_duration(kernel.MAX_WALL_TIME["train"]) == kernel.WALL_TIME


# --- GPUs and process settings ---------------------------------------------------------


def test_gpus_are_counted_with_nvidia_smi(
    kernel: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    listing = "GPU 0: Tesla T4 (UUID: GPU-a)\nGPU 1: Tesla T4 (UUID: GPU-b)\n"
    result = subprocess.CompletedProcess(["nvidia-smi"], 0, stdout=listing)
    monkeypatch.setattr(kernel.subprocess, "run", lambda *args, **kwargs: result)
    assert kernel.gpu_count() == 2

    def missing(*args: Any, **kwargs: Any) -> None:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(kernel.subprocess, "run", missing)
    assert kernel.gpu_count() == 0


def test_each_run_sees_one_gpu_and_the_second_stays_light(kernel: ModuleType) -> None:
    main = kernel.gpu_env(0)
    assert main["CUDA_VISIBLE_DEVICES"] == "0"
    assert main["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    second = kernel.gpu_env(1, second=True)
    assert second["CUDA_VISIBLE_DEVICES"] == "1"
    assert second["OMP_NUM_THREADS"] == second["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
    assert second["TORCHINDUCTOR_CACHE_DIR"] != main.get("TORCHINDUCTOR_CACHE_DIR")


def test_runs_write_to_their_own_directories(kernel: ModuleType, tmp_path: Path) -> None:
    command = kernel.train_command(Path("c.yaml"), Path("data"), tmp_path / "run-wide", "600s")
    assert command[command.index("--out") + 1] == str(tmp_path / "run-wide")
    assert command[command.index("--resume") + 1] == "auto"
    assert command[command.index("--max-wall-time") + 1] == "600s"


# --- supervision -----------------------------------------------------------------------


def test_the_kernel_waits_for_both_runs(kernel: ModuleType) -> None:
    main = start("import time; time.sleep(0.3)")
    second = start("import time; time.sleep(1.0)")
    started = time.monotonic()
    code = kernel.supervise(main, second, deadline=started + 60, grace=60, poll=0.05)
    assert code == 0
    assert second.returncode == 0  # waited for, not stopped
    assert time.monotonic() - started >= 0.9


def test_a_second_run_past_its_deadline_is_stopped_and_the_main_run_is_not(
    kernel: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    main = start("import time; time.sleep(1.5)")  # outlives the second run's deadline
    stubborn = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    second = start(stubborn)
    time.sleep(0.3)  # let it install its handler
    code = kernel.supervise(main, second, deadline=time.monotonic() + 0.2, grace=0.3, poll=0.05)
    assert code == 0
    assert main.returncode == 0  # never signalled
    assert second.returncode == -signal.SIGKILL  # ignored SIGTERM, so it was killed
    log = capsys.readouterr().out
    assert "run-wide is past its deadline: stopping it" in log
    assert "run-wide did not stop: killing it" in log


def test_the_main_runs_exit_code_is_the_kernels(kernel: ModuleType) -> None:
    main = start("raise SystemExit(3)")
    assert kernel.supervise(main, None, deadline=time.monotonic(), grace=1, poll=0.05) == 3


# --- the second run --------------------------------------------------------------------


def write_config(path: Path, **overrides: object) -> Path:
    settings = {
        "model": "tiny",
        "batch_size": 64,
        "max_steps": 30,
        "warmup_steps": 2,
        "log_every": 5,
        "val_every": 10,
        "val_batches": 2,
        "num_workers": 0,
        "compile": False,
        **overrides,
    }
    path.write_text("".join(f"{key}: {value}\n" for key, value in settings.items()))
    return path


@pytest.fixture
def setup(kernel: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A synthetic dataset, a tiny main-run config and a two-GPU machine; returns the data."""
    config = write_config(tmp_path / "train.yaml")
    monkeypatch.setitem(kernel.CONFIGS, "train", str(config))  # CHECKOUT / absolute = absolute
    monkeypatch.setattr(kernel, "gpu_count", lambda: 2)
    monkeypatch.setattr(kernel, "benchmark_second_model", lambda config, out: dict(FAST_SPEED))
    return make_dataset(tmp_path / "input" / "data" / "shards", positions=1200, shard_size=300)


def test_a_failing_second_run_does_not_affect_the_main_run(
    kernel: ModuleType,
    setup: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(kernel, "SECOND_MODEL", "no-such-model")  # its config fails to load
    assert kernel.train_mode(setup) == 0

    base, wide = tmp_path / "working" / "run-base", tmp_path / "working" / "run-wide"
    assert load_checkpoint(base / "checkpoints" / "step-000000030.pt").step == 30
    record = json.loads((wide / "epochs.json").read_text())
    assert record["epochs"] >= 1
    assert record["train_positions_per_second"] == FAST_SPEED["train_speed"]
    assert "no-such-model" in (wide / "train.log").read_text()  # the crash is in its own log
    log = capsys.readouterr().out
    assert "run-wide failed (exit code" in log
    assert "run-base ended with exit code 0" in log
    assert log.index("starting run-base") < log.index("starting run-wide")


def test_both_runs_train_in_their_own_directories(
    kernel: ModuleType,
    setup: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(kernel, "SECOND_MODEL", "tiny")  # trains in seconds on the CPU
    assert kernel.train_mode(setup) == 0

    working = tmp_path / "working"
    for name in ("run-base", "run-wide"):
        assert load_checkpoint(working / name / "checkpoints" / "step-000000030.pt").step == 30
    second = load_train_config(working / "run-wide" / "config.yaml")
    main = load_train_config(tmp_path / "train.yaml")
    epochs = json.loads((working / "run-wide" / "epochs.json").read_text())["epochs"]
    assert second == main.model_copy(update={"model": "tiny", "epochs": float(epochs)})
    log = capsys.readouterr().out
    assert "run-wide finished (exit code 0)" in log
    assert "output: " in log


def test_one_gpu_trains_only_the_main_run(
    kernel: ModuleType,
    setup: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(kernel, "gpu_count", lambda: 1)
    main = start("import time; time.sleep(30)")
    try:
        assert kernel.start_second_run(main, setup, time.monotonic() + 3600) is None
    finally:
        main.kill()
    assert "one GPU: only the main run trains" in capsys.readouterr().out


def test_no_second_run_when_the_main_run_fails_early(
    kernel: ModuleType, setup: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kernel, "benchmark_second_model", lambda *args: pytest.fail("benchmarked"))
    main = start("raise SystemExit(1)")
    assert kernel.start_second_run(main, setup, time.monotonic() + 3600) is None


def test_the_second_run_is_skipped_when_one_epoch_does_not_fit(
    kernel: ModuleType, setup: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slow = {**FAST_SPEED, "train_speed": 0.01}
    monkeypatch.setattr(kernel, "benchmark_second_model", lambda config, out: slow)
    main = start("pass")  # finished cleanly, as a completed run resumes
    assert kernel.start_second_run(main, setup, time.monotonic() + 3600) is None
    record = json.loads((tmp_path / "working" / "run-wide" / "epochs.json").read_text())
    assert record["epochs"] == 0
    assert "1 epoch takes about" in record["reason"]
    assert not (tmp_path / "working" / "run-wide" / "config.yaml").exists()


def test_a_resumed_second_run_keeps_its_config_and_skips_the_benchmark(
    kernel: ModuleType, setup: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kernel, "benchmark_second_model", lambda *args: pytest.fail("benchmarked"))
    wide = tmp_path / "working" / "run-wide"
    wide.mkdir()
    config = write_config(wide / "config.yaml", max_steps=12)
    commands: list[list[str]] = []

    def record_command(*args: Any) -> list[str]:
        commands.append([str(arg) for arg in args])
        return python("pass")

    monkeypatch.setattr(kernel, "train_command", record_command)
    second = kernel.start_second_run(start("pass"), setup, time.monotonic() + 3600)
    assert second is not None
    assert second.wait(timeout=60) == 0
    [(config_path, data, out, wall_time)] = commands
    assert (config_path, data, out) == (str(config), str(setup), str(wide))
    assert 3500 <= int(wall_time.removesuffix("s")) <= 3600
