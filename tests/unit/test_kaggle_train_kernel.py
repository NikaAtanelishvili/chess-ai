"""The Kaggle training kernel finds its inputs by globbing /kaggle/input and fails clearly."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

KERNEL = Path(__file__).parents[2] / "kaggle" / "train" / "train.py"


@pytest.fixture
def kernel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("kaggle_train_kernel", KERNEL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # defines functions only; main() is guarded
    monkeypatch.setattr(module, "INPUT", tmp_path)
    return module


def add_file(root: Path, relative: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    return path


def test_finds_the_single_manifest_wherever_it_is_mounted(
    kernel: ModuleType, tmp_path: Path
) -> None:
    manifest = add_file(tmp_path, "datasets/someone/chessnet-lichess/shards/manifest.json")
    add_file(tmp_path, "chessnet-run/run/runs.jsonl")  # a previous run has no manifest
    assert kernel.find_one("**/manifest.json", "manifests", required=True) == manifest.parent


def test_no_manifest_is_a_clear_error(kernel: ModuleType) -> None:
    with pytest.raises(SystemExit, match=r"No dataset manifests .*dataset_sources"):
        kernel.find_one("**/manifest.json", "dataset manifests", required=True)


def test_two_manifests_are_a_clear_error(kernel: ModuleType, tmp_path: Path) -> None:
    add_file(tmp_path, "a/shards/manifest.json")
    add_file(tmp_path, "b/shards/manifest.json")
    with pytest.raises(SystemExit, match="Found 2 manifests"):
        kernel.find_one("**/manifest.json", "manifests", required=True)


def test_a_previous_run_is_optional(kernel: ModuleType, tmp_path: Path) -> None:
    assert kernel.find_one("**/runs.jsonl", "previous runs", required=False) is None
    runs = add_file(tmp_path, "chessnet-run/run/runs.jsonl")
    assert kernel.find_one("**/runs.jsonl", "previous runs", required=False) == runs.parent


def test_an_unset_ref_stops_before_doing_anything(
    kernel: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kernel, "CHESSNET_REF", "<tag-or-commit-sha>")
    monkeypatch.setattr(kernel, "run", lambda *args, **kwargs: pytest.fail("must not run"))
    with pytest.raises(SystemExit, match="Set CHESSNET_REF"):
        kernel.main()


def test_an_unknown_mode_is_refused(kernel: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kernel, "CHESSNET_REF", "0" * 40)
    monkeypatch.setattr(kernel, "MODE", "full")
    with pytest.raises(SystemExit, match="MODE must be one of"):
        kernel.main()
