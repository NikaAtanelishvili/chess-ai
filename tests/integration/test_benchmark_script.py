"""The training benchmark script still runs (it is used in the first Kaggle GPU session)."""

import subprocess
import sys
from pathlib import Path

from helpers.datasets import make_dataset

SCRIPT = Path(__file__).parents[2] / "scripts" / "benchmark_train.py"


def test_benchmark_script_writes_a_report(tmp_path: Path) -> None:
    data = make_dataset(tmp_path / "shards", positions=600, shard_size=300)
    report = tmp_path / "benchmark.md"
    command = [
        sys.executable,
        str(SCRIPT),
        "--models",
        "tiny",
        "--batch-sizes",
        "16",
        "32",
        "--steps",
        "2",
        "--warmup",
        "1",
        "--device",
        "cpu",
        "--data",
        str(data),
        "--loader-batches",
        "2",
        "--workers",
        "0",
        "--out",
        str(report),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
    text = report.read_text(encoding="utf-8")
    assert "| tiny | 626,380 | 16 |" in text
    assert "| tiny | 626,380 | 32 |" in text
    assert "Data loader (0 workers, batch 32)" in text
