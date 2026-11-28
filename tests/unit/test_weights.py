"""Downloading the released weights: SHA-256 checked, nothing left behind on failure."""

import hashlib
import io
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from chessnet.cli import app
from chessnet.weights import RELEASE_FILES, RELEASE_URL, fetch, sha256_of

REPO = Path(__file__).parents[2]
PAYLOAD = b"released weights" * 1000
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


class Server:
    """An opener that serves ``payload`` and records the URLs asked for."""

    def __init__(self, payload: bytes = PAYLOAD) -> None:
        self.payload = payload
        self.urls: list[str] = []

    def __call__(self, url: str, timeout: float) -> io.BytesIO:
        self.urls.append(url)
        return io.BytesIO(self.payload)


def test_a_download_is_checked_and_moved_into_place(tmp_path: Path) -> None:
    server, lines = Server(), []
    target = tmp_path / "checkpoints" / "base-best.pt"
    fetch("base-best.pt", target, expected=DIGEST, opener=server, report=lines.append)
    assert target.read_bytes() == PAYLOAD
    assert server.urls == [RELEASE_URL.format(tag="v1.0.0", name="base-best.pt")]
    assert lines[-1].endswith("MB, SHA-256 verified")
    fetch("base-best.pt", target, expected=DIGEST, opener=server, report=lines.append)
    assert len(server.urls) == 1  # already there and verified: not downloaded again
    assert lines[-1].endswith("already there, SHA-256 verified")


def test_an_altered_download_is_refused_and_leaves_nothing(tmp_path: Path) -> None:
    target = tmp_path / "base-best.pt"
    with pytest.raises(ValueError, match="is not the released"):
        fetch("base-best.pt", target, expected=DIGEST, opener=Server(b"tampered"),
              report=lambda line: None)  # fmt: skip
    assert list(tmp_path.iterdir()) == []  # neither the file nor a .part


def test_a_different_file_is_replaced_only_with_force(tmp_path: Path) -> None:
    target = tmp_path / "base-best.pt"
    target.write_bytes(b"my own network")
    with pytest.raises(FileExistsError, match="--force"):
        fetch("base-best.pt", target, expected=DIGEST, opener=Server(), report=lambda line: None)
    assert target.read_bytes() == b"my own network"
    fetch("base-best.pt", target, expected=DIGEST, opener=Server(), force=True,
          report=lambda line: None)  # fmt: skip
    assert sha256_of(target) == DIGEST


def test_the_pinned_checkpoint_hash_is_the_one_in_the_training_docs() -> None:
    docs = (REPO / "docs" / "TRAINING.md").read_text(encoding="utf-8")
    row = next(line for line in docs.splitlines() if line.startswith("| `base-best.pt`"))
    assert re.search(r"`([0-9a-f]{64})`", row).group(1) == RELEASE_FILES["base-best.pt"]  # type: ignore[union-attr]


def test_cli_download_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Path, bool]] = []
    monkeypatch.setattr(
        "chessnet.weights.fetch",
        lambda name, destination, *, force, report: calls.append((name, destination, force)),
    )
    target = tmp_path / "w" / "base-best.pt"
    done = CliRunner().invoke(app, ["download-weights", "--out", str(target), "--onnx"])
    assert done.exit_code == 0, done.output
    assert calls == [
        ("base-best.pt", target, False),
        ("base-best.onnx", target.with_suffix(".onnx"), False),
    ]

    def refuse(name, destination, *, force, report):  # type: ignore[no-untyped-def]
        raise ValueError("the download's SHA-256 is wrong")

    monkeypatch.setattr("chessnet.weights.fetch", refuse)
    failed = CliRunner().invoke(app, ["download-weights", "--out", str(target)])
    assert failed.exit_code == 1
    assert "SHA-256 is wrong" in failed.output
