"""Kaggle script kernel (GPU, internet on): measure MCTS speed with the trained base network.

Uses the output of the training kernel (kernel_sources), finds run-base/best.pt in it, checks
its SHA-256 against the one recorded in docs/TRAINING.md, runs the GPU tests, and then
scripts/benchmark_search.py on the GPU and on the CPU. The reports land in
/kaggle/working (search-benchmark-cuda.md, search-benchmark-cpu.md and their JSON). It
takes about 10 minutes.

Before pushing, set CHESSNET_REF (a tag or commit SHA, never a branch). Push from the
repository root:

    uv run kaggle kernels push -p kaggle/search
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/NikaAtanelishvili/chess-ai"
CHESSNET_REF = "0641aba462e5b4db385e4998399fd8eadace6639"  # search code, pre-merge
CHECKPOINT_SHA256 = "370248af31086c8de72b9df0ef03a41a426e446a1fe31d6714fd0fe2306deb78"
INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
CHECKOUT = Path("/tmp/chessnet")  # outside /kaggle/working, so it is not saved as output
#: (device, leaf batch sizes): the GPU gets larger batches, the CPU the ones that pay there.
RUNS = (("cuda", ("1", "8", "16", "32", "64")), ("cpu", ("8", "16", "32")))


def run(*command: object, cwd: Path | None = None) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run([str(part) for part in command], check=True, cwd=cwd)


def find_checkpoint() -> Path:
    """The single run-base/best.pt under /kaggle/input (from the training kernel's output)."""
    matches = sorted(INPUT.glob("**/run-base/best.pt"))
    if len(matches) != 1:
        listing = "\n  ".join(map(str, matches)) or "(none)"
        sys.exit(
            f"Expected one run-base/best.pt under {INPUT}, found {len(matches)}:\n  {listing}\n"
            "Attach the training kernel's output in kernel-metadata.json (kernel_sources)."
        )
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if CHESSNET_REF.startswith("<"):
        sys.exit("Set CHESSNET_REF to a tag or commit SHA before pushing this kernel.")
    checkpoint = find_checkpoint()
    digest = sha256(checkpoint)
    print(f"checkpoint: {checkpoint}\nsha256: {digest}", flush=True)
    if digest != CHECKPOINT_SHA256:
        sys.exit(f"{checkpoint} is not the recorded base-best.pt (sha256 {CHECKPOINT_SHA256}).")

    run("git", "clone", "--quiet", REPOSITORY, CHECKOUT)
    run("git", "-C", CHECKOUT, "fetch", "--quiet", "origin", CHESSNET_REF)
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
    for device, batch_sizes in RUNS:
        run(
            sys.executable,
            "scripts/benchmark_search.py",
            "--checkpoint",
            checkpoint,
            "--device",
            device,
            "--batch-sizes",
            *batch_sizes,
            "--out",
            WORKING / f"search-benchmark-{device}.md",
            "--json",
            WORKING / f"search-benchmark-{device}.json",
            cwd=CHECKOUT,
        )


if __name__ == "__main__":
    main()
