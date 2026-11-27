"""Kaggle script kernel (internet on): the full evaluation suite of the trained networks.

Uses the training kernel's output (kernel_sources) for run-base/best.pt and run-wide/best.pt,
checks their SHA-256 against docs/TRAINING.md, installs Stockfish 16 (the official build),
downloads the Lichess puzzle database when puzzles are to run, and runs

    chessnet eval --suite full --weights base-best.pt --compare wide-best.pt \
        --workers 4 (GPU) or 2 (CPU) --max-wall-time 11h [--only ONLY]

Results (JSON, PGNs, the model card with the new table) land in /kaggle/working/eval. With
ONLY set, the results hold just those parts, to be merged into eval/results/ afterwards.

Before pushing, set CHESSNET_REF (a tag or commit SHA, never a branch) and ONLY. The
metadata asks for a CPU machine; for a T4, set "enable_gpu": true and "machine_shape":
"NvidiaTeslaT4" in kernel-metadata.json. Push from the repository root:

    uv run kaggle kernels push -p kaggle/eval
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

REPOSITORY = "https://github.com/NikaAtanelishvili/chess-ai"
CHESSNET_REF = "93f77c8b653238b43bfef45395c05a7a06adf74c"  # hang-proof evaluation, pre-merge
#: Parts of the suite to run (``chessnet eval --only``), or "" for the whole suite.
ONLY = "compare-equal-nodes"
#: The session limit is 12 h; the suite stops cleanly before it and keeps what finished.
MAX_WALL_TIME = "11h"
BASE_SHA256 = "370248af31086c8de72b9df0ef03a41a426e446a1fe31d6714fd0fe2306deb78"
WIDE_SHA256 = "7a3d263826103e6f85fa74c55f95b21ab988a8e7680b8eb1be94ac4e1cd4ca74"
CHECKPOINTS = {  # name in the results: (path in the training output, SHA-256)
    "base-best.pt": ("run-base/best.pt", BASE_SHA256),
    "wide-best.pt": ("run-wide/best.pt", WIDE_SHA256),
}
STOCKFISH_URL = (
    "https://github.com/official-stockfish/Stockfish/releases/download/sf_16/"
    "stockfish-ubuntu-x86-64-avx2.tar"
)
PUZZLES_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
GPU_WORKERS = 4  # two per T4
CPU_WORKERS = 2  # sharing torch's threads
INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
TOOLS = Path("/tmp/tools")  # outside /kaggle/working, so it is not saved as output
CHECKOUT = Path("/tmp/chessnet")


def run(*command: object, cwd: Path | None = None) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run([str(part) for part in command], check=True, cwd=cwd)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_checkpoints(destination: Path) -> dict[str, Path]:
    """Copy the two checkpoints out of the training output, under their result names."""
    destination.mkdir(parents=True, exist_ok=True)
    found = {}
    for name, (relative, expected) in CHECKPOINTS.items():
        matches = sorted(INPUT.glob(f"**/{relative}"))
        if len(matches) != 1:
            sys.exit(
                f"Expected one {relative} under {INPUT}, found {len(matches)}. Attach the "
                "training kernel's output in kernel-metadata.json (kernel_sources)."
            )
        digest = sha256(matches[0])
        if digest != expected:
            sys.exit(f"{matches[0]} has SHA-256 {digest}, not the recorded {expected}.")
        found[name] = Path(shutil.copy(matches[0], destination / name))
        print(f"{name}: {matches[0]} (sha256 ok)", flush=True)
    return found


def install_stockfish() -> Path | None:
    """The official Stockfish 16 build, or None (the suite then skips its matches)."""
    TOOLS.mkdir(parents=True, exist_ok=True)
    archive = TOOLS / "stockfish.tar"
    try:
        urllib.request.urlretrieve(STOCKFISH_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(TOOLS, filter="data")
    except (OSError, tarfile.TarError) as error:
        print(f"Stockfish download failed: {error}", flush=True)
        return None
    binaries = sorted(TOOLS.glob("stockfish*/stockfish-ubuntu-*"))
    if not binaries:
        print("No Stockfish binary in the archive", flush=True)
        return None
    binaries[0].chmod(0o755)
    return binaries[0]


def has_gpu() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True, check=False).returncode == 0


def needs_puzzles(only: str) -> bool:
    """The whole suite, or a selection with a puzzle run, needs the puzzle database."""
    return not only or any(part.strip().startswith("puzzles-") for part in only.split(","))


def eval_command(
    weights: dict[str, Path],
    *,
    workers: int,
    out: Path,
    card: Path,
    puzzles: Path | None,
    stockfish: Path | None,
    only: str = ONLY,
) -> list[object]:
    command: list[object] = [
        sys.executable, "-m", "chessnet", "eval", "--suite", "full",
        "--weights", weights["base-best.pt"], "--compare", weights["wide-best.pt"],
        "--workers", workers, "--device", "auto", "--out", out, "--model-card", card,
        "--max-wall-time", MAX_WALL_TIME,
    ]  # fmt: skip
    if puzzles is not None:
        command += ["--puzzles", puzzles]
    if stockfish is not None:
        command += ["--stockfish", stockfish]
    if only:
        command += ["--only", only]
    return command


def main() -> None:
    if CHESSNET_REF.startswith("<"):
        sys.exit("Set CHESSNET_REF to a tag or commit SHA before pushing this kernel.")
    weights = find_checkpoints(TOOLS / "weights")
    run("git", "clone", "--quiet", REPOSITORY, CHECKOUT)
    run("git", "-C", CHECKOUT, "fetch", "--quiet", "origin", CHESSNET_REF)
    run("git", "-C", CHECKOUT, "checkout", "--quiet", "FETCH_HEAD")
    run(sys.executable, "-m", "pip", "install", "--quiet", f"{CHECKOUT}[train]")
    stockfish = install_stockfish()
    puzzles = None
    if needs_puzzles(ONLY):
        puzzles = TOOLS / "lichess_db_puzzle.csv.zst"
        run("curl", "-sS", "-L", "--retry", "3", "-o", puzzles, PUZZLES_URL)
        run("curl", "-sSI", PUZZLES_URL)  # the database's date, for the record
    out = WORKING / "eval"
    out.mkdir(parents=True, exist_ok=True)
    card = out / "MODEL_CARD.md"
    shutil.copy(CHECKOUT / "docs" / "MODEL_CARD.md", card)
    workers = GPU_WORKERS if has_gpu() else CPU_WORKERS
    command = eval_command(
        weights, workers=workers, out=out / "results", card=card, puzzles=puzzles,
        stockfish=stockfish,
    )  # fmt: skip
    os.environ["PYTHONUNBUFFERED"] = "1"
    run(*command, cwd=CHECKOUT)


if __name__ == "__main__":
    main()
