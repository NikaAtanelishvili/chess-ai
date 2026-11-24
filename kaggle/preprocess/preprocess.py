"""Kaggle script kernel (CPU, internet on): build the chessnet training shards.

Before pushing, set CHESSNET_REF to a tag or commit SHA (never a moving branch) and replace
the ``<kaggle-username>`` placeholder in kernel-metadata.json. Then, from the repository root:

    kaggle kernels push -p kaggle/preprocess

The shards and manifest.json land in /kaggle/working/shards, which becomes the kernel output.
See docs/TRAINING.md for turning that output into a Kaggle Dataset.
"""

import subprocess
import sys

REPOSITORY = "https://github.com/NikaAtanelishvili/chess-ai"
CHESSNET_REF = "873a1af741407d94bcd2c20a0de915574caa6f96"  # v0.1.0
SOURCE = "https://database.lichess.org/standard/lichess_db_standard_rated_2026-10.pgn.zst"
CONFIG = "configs/data/lichess_2000plus.yaml"
CHECKOUT = "/tmp/chessnet"  # outside /kaggle/working, so it is not saved as output
OUTPUT = "/kaggle/working/shards"


def run(*command: str) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if CHESSNET_REF.startswith("<"):
    sys.exit("Set CHESSNET_REF to a tag or commit SHA before pushing this kernel.")

run("git", "clone", "--quiet", REPOSITORY, CHECKOUT)
run("git", "-C", CHECKOUT, "checkout", "--quiet", CHESSNET_REF)
run(sys.executable, "-m", "pip", "install", "--quiet", CHECKOUT)
run(
    sys.executable,
    "-m",
    "chessnet",
    "preprocess",
    "--config",
    f"{CHECKOUT}/{CONFIG}",
    "--source",
    SOURCE,
    "--out",
    OUTPUT,
)
