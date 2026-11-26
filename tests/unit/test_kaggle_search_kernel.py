"""The Kaggle search kernel finds the trained checkpoint, checks it, and fails clearly."""

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).parents[2]
KERNEL = REPO / "kaggle" / "search" / "search.py"


@pytest.fixture
def kernel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("kaggle_search_kernel", KERNEL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # defines functions only; main() is guarded
    monkeypatch.setattr(module, "INPUT", tmp_path)
    return module


def add_file(root: Path, relative: str, content: bytes = b"weights") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_finds_the_base_checkpoint_in_the_training_output(
    kernel: ModuleType, tmp_path: Path
) -> None:
    best = add_file(tmp_path, "chessnet-train/run-base/best.pt")
    add_file(tmp_path, "chessnet-train/run-wide/best.pt")  # the other run is ignored
    add_file(tmp_path, "chessnet-train/run-base/checkpoints/step-000187860.pt")
    assert kernel.find_checkpoint() == best


@pytest.mark.parametrize("count", [0, 2])
def test_anything_but_one_checkpoint_is_a_clear_error(
    kernel: ModuleType, tmp_path: Path, count: int
) -> None:
    for index in range(count):
        add_file(tmp_path, f"output-{index}/run-base/best.pt")
    with pytest.raises(SystemExit, match=rf"found {count}:[\s\S]*kernel_sources"):
        kernel.find_checkpoint()


def test_a_different_checkpoint_is_refused(
    kernel: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_file(tmp_path, "chessnet-train/run-base/best.pt", b"not the recorded weights")
    monkeypatch.setattr(kernel, "CHESSNET_REF", "0" * 40)
    monkeypatch.setattr(kernel, "run", lambda *args, **kwargs: pytest.fail("must not run"))
    with pytest.raises(SystemExit, match=r"is not the recorded base-best\.pt"):
        kernel.main()


def test_an_unset_ref_stops_before_doing_anything(
    kernel: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kernel, "CHESSNET_REF", "<tag-or-commit-sha>")
    monkeypatch.setattr(kernel, "run", lambda *args, **kwargs: pytest.fail("must not run"))
    with pytest.raises(SystemExit, match="Set CHESSNET_REF"):
        kernel.main()


def test_the_expected_hash_is_the_one_in_the_training_docs(kernel: ModuleType) -> None:
    docs = (REPO / "docs" / "TRAINING.md").read_text(encoding="utf-8")
    row = next(line for line in docs.splitlines() if line.startswith("| `base-best.pt`"))
    assert re.search(r"`([0-9a-f]{64})`", row).group(1) == kernel.CHECKPOINT_SHA256  # type: ignore[union-attr]


def test_sha256(kernel: ModuleType, tmp_path: Path) -> None:
    path = add_file(tmp_path, "file", b"abc")
    assert kernel.sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
