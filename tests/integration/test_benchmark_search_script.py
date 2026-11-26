"""The search benchmark script runs (the Kaggle search kernel uses it)."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "benchmark_search.py"


def test_benchmark_search_writes_a_report(tmp_path: Path) -> None:
    report, measured = tmp_path / "search.md", tmp_path / "search.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--model",
        "tiny",
        "--device",
        "cpu",
        "--simulations",
        "24",
        "--batch-sizes",
        "1",
        "4",
        "--out",
        str(report),
        "--json",
        str(measured),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
    text = report.read_text(encoding="utf-8")
    assert "| 1 | " in text
    assert "| 4 | " in text
    assert "- Lucena rook endgame: " in text
    data = json.loads(measured.read_text(encoding="utf-8"))
    assert [row["batch_size"] for row in data["summary"]] == [1, 4]
    assert len(data["rows"]) == 2 * 5
    assert all(row["nodes_per_second"] > 0 for row in data["rows"])
    assert all(row["mean_batch"] <= row["batch_size"] for row in data["rows"])
