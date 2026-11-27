"""The Kaggle eval kernel finds both checkpoints, checks them, and fails clearly."""

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).parents[2]
KERNEL = REPO / "kaggle" / "eval" / "eval.py"


@pytest.fixture
def kernel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("kaggle_eval_kernel", KERNEL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # defines functions only; main() is guarded
    monkeypatch.setattr(module, "INPUT", tmp_path / "input")
    return module


def add_file(root: Path, relative: str, content: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_both_checkpoints_are_found_checked_and_renamed(
    kernel: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_file(tmp_path, "input/chessnet-train/run-base/best.pt", b"base")
    add_file(tmp_path, "input/chessnet-train/run-wide/best.pt", b"wide")
    monkeypatch.setattr(
        kernel,
        "CHECKPOINTS",
        {
            "base-best.pt": (
                "run-base/best.pt",
                kernel.sha256(tmp_path / "input/chessnet-train/run-base/best.pt"),
            ),
            "wide-best.pt": (
                "run-wide/best.pt",
                kernel.sha256(tmp_path / "input/chessnet-train/run-wide/best.pt"),
            ),
        },
    )
    found = kernel.find_checkpoints(tmp_path / "weights")
    assert found["base-best.pt"].read_bytes() == b"base"
    assert found["wide-best.pt"].name == "wide-best.pt"


def test_a_missing_or_different_checkpoint_stops_the_kernel(
    kernel: ModuleType, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="kernel_sources"):
        kernel.find_checkpoints(tmp_path / "weights")
    add_file(tmp_path, "input/chessnet-train/run-base/best.pt", b"other weights")
    with pytest.raises(SystemExit, match="not the recorded"):
        kernel.find_checkpoints(tmp_path / "weights")


def test_the_recorded_hashes_are_the_ones_in_the_training_docs(kernel: ModuleType) -> None:
    docs = (REPO / "docs" / "TRAINING.md").read_text(encoding="utf-8")
    for name, (_, digest) in kernel.CHECKPOINTS.items():
        row = next(line for line in docs.splitlines() if line.startswith(f"| `{name}`"))
        assert re.search(r"`([0-9a-f]{64})`", row).group(1) == digest  # type: ignore[union-attr]


def test_an_unset_ref_stops_before_doing_anything(
    kernel: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kernel, "CHESSNET_REF", "<tag-or-commit-sha>")
    monkeypatch.setattr(kernel, "run", lambda *args, **kwargs: pytest.fail("must not run"))
    with pytest.raises(SystemExit, match="Set CHESSNET_REF"):
        kernel.main()


def test_the_command_is_bounded_in_time_and_runs_only_the_chosen_parts(
    kernel: ModuleType,
) -> None:
    weights = {"base-best.pt": Path("b.pt"), "wide-best.pt": Path("w.pt")}
    command = [str(part) for part in kernel.eval_command(
        weights, workers=2, out=Path("out"), card=Path("card.md"), puzzles=None,
        stockfish=Path("sf"), only="compare-equal-nodes",
    )]  # fmt: skip
    assert command[command.index("--max-wall-time") + 1] == kernel.MAX_WALL_TIME
    assert command[command.index("--heartbeat") + 1] == "60"
    assert command[command.index("--only") + 1] == "compare-equal-nodes"
    assert "--puzzles" not in command
    whole = kernel.eval_command(weights, workers=4, out=Path("out"), card=Path("card.md"),
                                puzzles=Path("p.zst"), stockfish=None, only="")  # fmt: skip
    assert "--only" not in whole
    assert "--puzzles" in whole
    assert kernel.needs_puzzles("")
    assert kernel.needs_puzzles("a, puzzles-policy")
    assert not kernel.needs_puzzles("compare-equal-nodes")


def test_the_chosen_parts_exist_in_the_full_suite(kernel: ModuleType) -> None:
    from chessnet.eval.suite import plan_suite, puzzle_slug

    plan = plan_suite("full", "base-best.pt", compare="wide-best.pt")  # without Stockfish
    known = {m.slug for m in plan.matches} | {puzzle_slug(p) for p in plan.puzzle_players}
    known |= {"mcts800-vs-stockfish-elo", "mcts800-vs-stockfish-skill0"}
    assert {part.strip() for part in kernel.ONLY.split(",") if part.strip()} <= known


def test_the_metadata_matches_the_machine_the_parts_are_sized_for(kernel: ModuleType) -> None:
    import json

    metadata = json.loads((KERNEL.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["enable_gpu"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"  # T4 x2
    assert metadata["kernel_sources"] == ["nikaatanelishvili/chessnet-train"]
