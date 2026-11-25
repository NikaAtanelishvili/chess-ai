"""Kaggle script kernel (GPU, internet on): check the GPU code path, then train chessnet.

MODE "smoke" (first session): GPU tests, the training throughput benchmark, and a short run of
the tiny model on about 1M positions. Its output (benchmark.md) set the main run's settings.

MODE "train": GPU tests, then two independent training runs in separate processes:

- run-base/, the main run: the base model on GPU 0 with configs/train/kaggle_t4.yaml. It
  starts first, and nothing the second run does can stop or slow it.
- run-wide/, the second run, only on a machine with two GPUs: the wide model on GPU 1 with
  the same settings. Once the main run is training, a short benchmark on GPU 1 measures the
  wide model, and its epochs become the most that leave 1.5 h of its wall time
  (run-wide/epochs.json records the speed and the choice, or why the run was skipped). It
  runs at the lowest CPU priority and logs to run-wide/train.log.

Both runs stop with a checkpoint 11 h 15 min after the main run starts, before Kaggle's
12-hour limit, and the kernel exits once both have ended. If a previous session's output is
attached as an input dataset, each run continues from the newest checkpoint in its own
directory.

Before pushing, set CHESSNET_REF (a tag or commit SHA, never a branch) and MODE. Push from the
repository root:

    uv run kaggle kernels push -p kaggle/train
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY = "https://github.com/NikaAtanelishvili/chess-ai"
CHESSNET_REF = "6a60990c4f6be1be2a51fdf58c3f58aa36387503"  # main: training merged (#10)
MODE = "train"  # "smoke" or "train"

CONFIGS = {"smoke": "configs/train/smoke.yaml", "train": "configs/train/kaggle_t4.yaml"}
MAX_WALL_TIME = {"smoke": "1h", "train": "11h15m"}
INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
CHECKOUT = Path("/tmp/chessnet")  # outside /kaggle/working, so it is not saved as output

MAIN_RUN, SECOND_RUN = "run-base", "run-wide"
SECOND_MODEL = "wide"
WALL_TIME = 11 * 3600 + 15 * 60  # MAX_WALL_TIME["train"] in seconds, from the main run's start
SECOND_MARGIN = 1.5 * 3600  # the second run's epochs leave this much of its wall time
TRAINING_TIMEOUT = 30 * 60  # the main run must log progress by then for the second to start
BENCHMARK_TIMEOUT = 15 * 60
STOP_GRACE = 10 * 60  # after its deadline the second run gets SIGTERM, then SIGKILL as late
POLL = 30.0  # seconds between checks while waiting
OUTPUT_LIMIT_GB = 20  # Kaggle keeps at most this much of /kaggle/working
LOW_PRIORITY = ("nice", "-n", "19")


def log(message: str) -> None:
    print(f"[kernel] {message}", flush=True)


def run(*command: object, cwd: Path | None = None) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run([str(part) for part in command], check=True, cwd=cwd)


def find_one(pattern: str, what: str, *, required: bool) -> Path | None:
    """The directory of the single file matching ``pattern`` under /kaggle/input."""
    matches = sorted(INPUT.glob(pattern))
    if len(matches) > 1:
        listing = "\n  ".join(map(str, matches))
        sys.exit(f"Found {len(matches)} {what} under {INPUT}; attach only one:\n  {listing}")
    if not matches:
        if required:
            sys.exit(
                f"No {what} under {INPUT}. Attach the dataset in kernel-metadata.json "
                "(dataset_sources) and push again."
            )
        return None
    return matches[0].parent


def gpu_count() -> int:
    """GPUs that nvidia-smi lists; 0 if it fails."""
    try:
        listing = subprocess.run(
            ["nvidia-smi", "--list-gpus"], capture_output=True, text=True, check=True, timeout=60
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(line.startswith("GPU ") for line in listing.splitlines())


def gpu_env(index: int, *, second: bool = False) -> dict[str, str]:
    """Environment for a process that sees only GPU ``index`` (nvidia-smi's numbering)."""
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(index)}
    if second:  # little CPU and its own compiler caches, so the main run never waits on it
        env |= {
            "OMP_NUM_THREADS": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor-second",
            "TRITON_CACHE_DIR": "/tmp/triton-second",
        }
    return env


def train_command(config: Path, data: Path, out: Path, max_wall_time: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "chessnet",
        "train",
        "--config",
        str(config),
        "--data",
        str(data),
        "--out",
        str(out),
        "--resume",
        "auto",
        "--max-wall-time",
        max_wall_time,
    ]


def restore_previous_runs() -> None:
    """Copy each run directory of an attached previous session into /kaggle/working."""
    for name in (MAIN_RUN, SECOND_RUN):
        what = f"previous {name} runs (runs.jsonl)"
        previous = find_one(f"**/{name}/runs.jsonl", what, required=False)
        target = WORKING / name
        if previous is not None and not target.exists():
            log(f"continuing {name} from {previous}")
            shutil.copytree(previous, target)


def wait_until_training(
    main: subprocess.Popen[bytes], metrics: Path, *, timeout: float, poll: float
) -> bool:
    """Whether the main run got going: it logged training progress or finished cleanly."""
    size = metrics.stat().st_size if metrics.exists() else 0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if main.poll() is not None:
            return main.returncode == 0
        if metrics.exists() and metrics.stat().st_size > size:
            return True
        time.sleep(poll)
    return False


def on_off(value: bool) -> str:
    return "on" if value else "off"


def benchmark_second_model(config: Any, out: Path) -> dict[str, Any] | None:
    """Training speed of the second run's model on GPU 1; ``None`` if the benchmark failed."""
    report = out / "benchmark.json"
    command = [
        *LOW_PRIORITY,
        sys.executable,
        "scripts/benchmark_train.py",
        "--models",
        config.model,
        "--batch-sizes",
        str(config.batch_size),
        "--compile",
        on_off(config.compile),
        "--channels-last",
        on_off(config.channels_last),
        "--device",
        "cuda",
        "--out",
        str(out / "benchmark.md"),
        "--json",
        str(report),
    ]
    with (out / "benchmark.log").open("a", encoding="utf-8") as output:
        try:
            subprocess.run(
                command,
                cwd=CHECKOUT,
                env=gpu_env(1, second=True),
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=BENCHMARK_TIMEOUT,
                check=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            log(f"the {config.model} benchmark failed: {error}")
            return None
    results: list[dict[str, Any]] = json.loads(report.read_text(encoding="utf-8"))["results"]
    return results[0] if results else None


def choose_epochs(
    config: Any, speed: dict[str, Any], data: Path, budget: float
) -> tuple[int, dict[str, Any]]:
    """The most epochs that leave SECOND_MARGIN of ``budget`` seconds, and the record of why."""
    from chessnet.train.budget import RunCosts, estimate_seconds, max_epochs, steps_per_epoch

    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    shards = manifest["splits"]["train"]["shards"]
    costs = RunCosts(
        steps_per_epoch=steps_per_epoch(
            (shard["positions"] for shard in shards), config.batch_size
        ),
        batch_size=config.batch_size,
        train_speed=speed["train_speed"],
        eval_speed=speed["eval_speed"],
        val_every=config.val_every,
        val_positions=config.val_batches * config.batch_size,
        checkpoint_minutes=config.checkpoint_minutes,
        checkpoint_seconds=2.0,  # about 90 MB for the wide model
        startup_seconds=60.0 + speed["first_step"],  # imports and workers, then compilation
    )
    epochs = max_epochs(budget, SECOND_MARGIN, costs)
    record = {
        "model": config.model,
        "batch_size": config.batch_size,
        "compile": config.compile,
        "channels_last": config.channels_last,
        "train_positions_per_second": speed["train_speed"],
        "eval_positions_per_second": speed["eval_speed"],
        "first_step_seconds": speed["first_step"],
        "steps_per_epoch": costs.steps_per_epoch,
        "wall_time_seconds": round(budget),
        "margin_seconds": SECOND_MARGIN,
        "epochs": epochs,
        "estimated_seconds": round(estimate_seconds(max(epochs, 1), costs)),
    }
    return epochs, record


def start_second_run(
    main: subprocess.Popen[bytes], data: Path, deadline: float
) -> subprocess.Popen[bytes] | None:
    """Start the second run on GPU 1, or return ``None`` (with the reason logged) to skip it."""
    import yaml  # installed with chessnet

    from chessnet.config import load_train_config

    if gpu_count() < 2:
        log("one GPU: only the main run trains")
        return None
    metrics = WORKING / MAIN_RUN / "metrics.csv"
    if not wait_until_training(main, metrics, timeout=TRAINING_TIMEOUT, poll=POLL):
        log(f"{MAIN_RUN} is not training, so {SECOND_RUN} does not start")
        return None
    out = WORKING / SECOND_RUN
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.yaml"
    if config_path.exists():  # chosen in an earlier session; the run's schedule continues
        log(f"continuing {SECOND_RUN} with its recorded {config_path.name}")
    else:
        base = load_train_config(CHECKOUT / CONFIGS["train"])
        config = base.model_copy(update={"model": SECOND_MODEL})
        speed = benchmark_second_model(config, out)
        if speed is None:
            return skip_second_run(out, {"reason": "the benchmark failed; see benchmark.log"})
        epochs, record = choose_epochs(config, speed, data, deadline - time.monotonic())
        if epochs < 1:
            hours = record["estimated_seconds"] / 3600
            reason = f"1 epoch takes about {hours:.1f} h, more than the time left minus 1.5 h"
            return skip_second_run(out, {**record, "reason": reason})
        (out / "epochs.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        chosen = config.model_copy(update={"epochs": float(epochs)}).model_dump()
        config_path.write_text(yaml.safe_dump(chosen, sort_keys=False), encoding="utf-8")
        log(
            f"{SECOND_RUN}: {speed['train_speed']:,.0f} positions/s, {epochs} epochs "
            f"(estimated {record['estimated_seconds'] / 3600:.2f} h)"
        )
    remaining = int(deadline - time.monotonic())
    command = [*LOW_PRIORITY, *train_command(config_path, data, out, f"{remaining}s")]
    log(f"starting {SECOND_RUN} on GPU 1; its log is {SECOND_RUN}/train.log")
    with (out / "train.log").open("ab") as output:
        return subprocess.Popen(
            command, env=gpu_env(1, second=True), stdout=output, stderr=subprocess.STDOUT
        )


def skip_second_run(out: Path, record: dict[str, Any]) -> None:
    (out / "epochs.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    log(f"{SECOND_RUN} skipped: {record['reason']}")


def supervise(
    main: subprocess.Popen[bytes],
    second: subprocess.Popen[bytes] | None,
    *,
    deadline: float,
    grace: float,
    poll: float,
) -> int:
    """Wait until both runs have ended and return the main run's exit code.

    The main run is only ever waited on. A second run still going at ``deadline`` gets
    SIGTERM (it saves a checkpoint and stops), and SIGKILL ``grace`` seconds later.
    """
    stopping_since: float | None = None
    main_done, second_done = False, second is None
    while True:
        if not main_done and main.poll() is not None:
            main_done = True
            log(f"{MAIN_RUN} ended with exit code {main.returncode}")
        if second is not None and not second_done:
            if second.poll() is not None:
                second_done = True
                outcome = "finished" if second.returncode == 0 else "failed"
                log(f"{SECOND_RUN} {outcome} (exit code {second.returncode})")
            elif time.monotonic() >= deadline:
                if stopping_since is None:
                    log(f"{SECOND_RUN} is past its deadline: stopping it")
                    second.terminate()
                    stopping_since = time.monotonic()
                elif time.monotonic() >= stopping_since + grace:
                    log(f"{SECOND_RUN} did not stop: killing it")
                    second.kill()
        if main_done and second_done:
            return main.returncode
        time.sleep(poll)


def report_output_size() -> None:
    total = 0
    for name in (MAIN_RUN, SECOND_RUN):
        files = [path for path in (WORKING / name).rglob("*") if path.is_file()]
        size = sum(path.stat().st_size for path in files)
        total += size
        log(f"{name}: {size / 1e9:.2f} GB")
    log(f"output: {total / 1e9:.2f} GB of Kaggle's {OUTPUT_LIMIT_GB} GB")


def train_mode(data: Path) -> int:
    """Both runs of a training session; returns the main run's exit code."""
    restore_previous_runs()
    config = CHECKOUT / CONFIGS["train"]
    command = train_command(config, data, WORKING / MAIN_RUN, MAX_WALL_TIME["train"])
    log(f"starting {MAIN_RUN} on GPU 0")
    main_run = subprocess.Popen(command, env=gpu_env(0))
    deadline = time.monotonic() + WALL_TIME  # both runs stop by then
    try:
        second = start_second_run(main_run, data, deadline)
    except Exception as error:  # nothing about the second run may end the kernel early
        log(f"{SECOND_RUN} not started: {type(error).__name__}: {error}")
        second = None
    code = supervise(main_run, second, deadline=deadline + STOP_GRACE, grace=STOP_GRACE, poll=POLL)
    try:
        report_output_size()
    except OSError as error:  # a report must not change the session's result
        log(f"could not measure the output: {error}")
    return code


def main() -> None:
    if CHESSNET_REF.startswith("<"):
        sys.exit("Set CHESSNET_REF to a tag or commit SHA before pushing this kernel.")
    if MODE not in CONFIGS:
        sys.exit(f"MODE must be one of {sorted(CONFIGS)}, not {MODE!r}.")

    run("git", "clone", "--quiet", REPOSITORY, CHECKOUT)
    run("git", "-C", CHECKOUT, "fetch", "--quiet", "origin", CHESSNET_REF)  # tags and SHAs alike
    run("git", "-C", CHECKOUT, "checkout", "--quiet", "FETCH_HEAD")
    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        f"{CHECKOUT}[train]",
        "pytest",
        "hypothesis",
    )
    run("nvidia-smi")
    # Fails (instead of skipping) when no GPU is attached or the GPU unpacking path is wrong.
    run(
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "gpu",
        "--require-gpu",
        "-q",
        "-p",
        "no:cacheprovider",
        cwd=CHECKOUT,
    )

    data = find_one("**/manifest.json", "dataset manifests (manifest.json)", required=True)
    print(f"dataset: {data}", flush=True)
    if MODE == "smoke":
        run(
            sys.executable,
            "scripts/benchmark_train.py",
            "--data",
            data,
            "--out",
            WORKING / "benchmark.md",
            cwd=CHECKOUT,
        )
        run(*train_command(CHECKOUT / CONFIGS[MODE], data, WORKING / "smoke", MAX_WALL_TIME[MODE]))
        return
    sys.exit(train_mode(data))


if __name__ == "__main__":
    main()
