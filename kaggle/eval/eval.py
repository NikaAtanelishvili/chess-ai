"""Kaggle script kernel (internet on): evaluation of the trained networks.

TASK "suite" runs parts of the full evaluation suite (ONLY); TASK "ladder" rates MCTS 800
on Stockfish's UCI_Elo scale with a ladder, then plays the playout-scaling curve
(`chessnet ladder`, LADDER_ARGS).

Uses the training kernel's output (kernel_sources) for run-base/best.pt and run-wide/best.pt,
checks their SHA-256 against docs/TRAINING.md, installs Stockfish 16 (the official build),
downloads the Lichess puzzle database when puzzles are to run, and runs

    chessnet eval --suite full --weights base-best.pt --compare wide-best.pt \
        --workers 4 (GPU) or 2 (CPU) --max-wall-time MAX_WALL_TIME --heartbeat 60 \
        [--only ONLY]

Matches with a time limit per move play one game per GPU. Results (JSON, PGNs, the model
card with the new table) land in /kaggle/working/eval. With ONLY set, the results hold just
those parts; merge them into eval/results/ with `chessnet eval --merge-results`.

Before pushing, set CHESSNET_REF (a tag or commit SHA, never a branch), ONLY and
MAX_WALL_TIME. The metadata asks for the T4 x2 machine; for a CPU machine, set
"enable_gpu": false and drop "machine_shape" in kernel-metadata.json. Push from the
repository root:

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
CHESSNET_REF = "9d55d22af4c4ac643cf1734d0d69c33e371ad4e2"  # the ladder, pre-merge
#: "suite" (chessnet eval, with ONLY) or "ladder" (chessnet ladder, with LADDER_ARGS).
TASK = "ladder"
#: The ladder: UCI_Elo levels, games per level, the stop, then the playout-scaling curve.
LADDER_ARGS = (
    "--levels", "1500,1700,1900,2100,2300", "--games", "40", "--stop-below", "0.25",
    "--simulations", "800", "--scaling", "50,100,200,400,800,1600", "--scaling-games", "32",
)  # fmt: skip
#: Parts of the suite to run (``chessnet eval --only``), or "" for the whole suite. The
#: policy's puzzle run is repeated here so that both puzzle runs score the same sample.
ONLY = (
    "mcts800-vs-stockfish-skill0,compare-equal-nodes,compare-equal-time,"
    "puzzles-policy,puzzles-mcts-800"
)
#: The run stops cleanly at this wall time and keeps what finished (ladder estimate: 1.7 to
#: 2.4 h).
MAX_WALL_TIME = "3h30m"
HEARTBEAT_SECONDS = 60
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
        "--max-wall-time", MAX_WALL_TIME, "--heartbeat", HEARTBEAT_SECONDS,
    ]  # fmt: skip
    if puzzles is not None:
        command += ["--puzzles", puzzles]
    if stockfish is not None:
        command += ["--stockfish", stockfish]
    if only:
        command += ["--only", only]
    return command


def ladder_command(
    weights: dict[str, Path], *, workers: int, out: Path, card: Path, stockfish: Path
) -> list[object]:
    return [
        sys.executable, "-m", "chessnet", "ladder", "--weights", weights["base-best.pt"],
        "--stockfish", stockfish, "--workers", workers, "--device", "auto", "--out", out,
        "--model-card", card, "--max-wall-time", MAX_WALL_TIME,
        "--heartbeat", HEARTBEAT_SECONDS, *LADDER_ARGS,
    ]  # fmt: skip


def main() -> None:
    if CHESSNET_REF.startswith("<"):
        sys.exit("Set CHESSNET_REF to a tag or commit SHA before pushing this kernel.")
    if TASK not in ("suite", "ladder"):
        sys.exit(f"TASK is 'suite' or 'ladder', not {TASK!r}.")
    weights = find_checkpoints(TOOLS / "weights")
    run("git", "clone", "--quiet", REPOSITORY, CHECKOUT)
    run("git", "-C", CHECKOUT, "fetch", "--quiet", "origin", CHESSNET_REF)
    run("git", "-C", CHECKOUT, "checkout", "--quiet", "FETCH_HEAD")
    run(sys.executable, "-m", "pip", "install", "--quiet", f"{CHECKOUT}[train]")
    stockfish = install_stockfish()
    if TASK == "ladder" and stockfish is None:
        sys.exit("The ladder needs Stockfish, and it could not be installed.")
    puzzles = None
    if TASK == "suite" and needs_puzzles(ONLY):
        puzzles = TOOLS / "lichess_db_puzzle.csv.zst"
        run("curl", "-sS", "-L", "--retry", "3", "-o", puzzles, PUZZLES_URL)
        run("curl", "-sSI", PUZZLES_URL)  # the database's date, for the record
    out = WORKING / "eval"
    out.mkdir(parents=True, exist_ok=True)
    card = out / "MODEL_CARD.md"
    shutil.copy(CHECKOUT / "docs" / "MODEL_CARD.md", card)
    workers = GPU_WORKERS if has_gpu() else CPU_WORKERS
    if TASK == "ladder":
        assert stockfish is not None
        command = ladder_command(
            weights, workers=workers, out=out / "results", card=card, stockfish=stockfish
        )
    else:
        command = eval_command(
            weights, workers=workers, out=out / "results", card=card, puzzles=puzzles,
            stockfish=stockfish,
        )  # fmt: skip
    os.environ["PYTHONUNBUFFERED"] = "1"
    run(*command, cwd=CHECKOUT)


if __name__ == "__main__":
    main()
