"""Kaggle script kernel (GPU, internet on): check the GPU code path, then train chessnet.

MODE "smoke" (first session): GPU tests, the training throughput benchmark, and a short run of
the tiny model on about 1M positions. Its output (benchmark.md) decides the main run's
settings.

MODE "train": GPU tests, then the main run with the base model. If a previous session's output
is attached as an input dataset, the run continues from its newest checkpoint. The run stops
with a checkpoint after 11 h 15 min, before Kaggle's 12-hour limit.

Before pushing, set CHESSNET_REF (a tag or commit SHA, never a branch) and MODE. Push from the
repository root:

    uv run kaggle kernels push -p kaggle/train
"""

import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/NikaAtanelishvili/chess-ai"
CHESSNET_REF = "ecc82245539d947b6dd6900afa4874fe7ace1dbd"  # training code, pre-merge
MODE = "smoke"  # "smoke" or "train"

CONFIGS = {"smoke": "configs/train/smoke.yaml", "train": "configs/train/kaggle_t4.yaml"}
MAX_WALL_TIME = {"smoke": "1h", "train": "11h15m"}
INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
CHECKOUT = Path("/tmp/chessnet")  # outside /kaggle/working, so it is not saved as output


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
    train = [
        sys.executable,
        "-m",
        "chessnet",
        "train",
        "--config",
        CHECKOUT / CONFIGS[MODE],
        "--data",
        data,
    ]
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
        run(*train, "--out", WORKING / "smoke", "--max-wall-time", MAX_WALL_TIME[MODE])
        return

    run_dir = WORKING / "run"
    previous = find_one("**/runs.jsonl", "previous runs (runs.jsonl)", required=False)
    if previous is not None and not run_dir.exists():
        print(f"continuing the run in {previous}", flush=True)
        shutil.copytree(previous, run_dir)
    run(*train, "--out", run_dir, "--resume", "auto", "--max-wall-time", MAX_WALL_TIME[MODE])


if __name__ == "__main__":
    main()
