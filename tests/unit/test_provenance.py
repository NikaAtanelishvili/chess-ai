import json
import re
from importlib import metadata
from pathlib import Path

import pytest

from chessnet import provenance


def test_git_commit_of_this_checkout() -> None:
    commit = provenance.git_commit()
    assert commit is not None
    assert re.fullmatch(r"[0-9a-f]{40}(-dirty)?", commit)


class _FakeDistribution:
    def __init__(self, record: str | None) -> None:
        self._record = record

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        return self._record


def test_commit_comes_from_a_vcs_install_record(monkeypatch: pytest.MonkeyPatch) -> None:
    record = json.dumps(
        {"url": "https://example.invalid/repo", "vcs_info": {"commit_id": "abc123"}}
    )
    monkeypatch.setattr(provenance.metadata, "distribution", lambda _: _FakeDistribution(record))
    assert provenance.git_commit() == "abc123"


@pytest.mark.parametrize(
    "record",
    [None, json.dumps({"url": "file:///src", "dir_info": {"editable": True}})],
)
def test_without_vcs_record_git_is_asked(
    monkeypatch: pytest.MonkeyPatch, record: str | None
) -> None:
    monkeypatch.setattr(provenance.metadata, "distribution", lambda _: _FakeDistribution(record))
    monkeypatch.setattr(provenance, "_commit_from_git", lambda directory: "from-git")
    assert provenance.git_commit() == "from-git"


def test_missing_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> None:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(provenance.metadata, "distribution", missing)
    assert provenance._commit_from_install_record() is None


def test_outside_a_repository_there_is_no_commit(tmp_path: Path) -> None:
    assert provenance._commit_from_git(tmp_path) is None
