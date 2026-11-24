"""Where the running code came from, for dataset manifests and checkpoints."""

from __future__ import annotations

import json
import subprocess
from importlib import metadata
from pathlib import Path


def git_commit() -> str | None:
    """Commit of the running chessnet code, or ``None`` if it cannot be determined.

    For ``pip install git+...@<ref>`` installs (e.g. on Kaggle) the commit comes from the
    installer's record. Otherwise git is asked in the source tree; ``-dirty`` is appended when
    tracked files have uncommitted changes.
    """
    return _commit_from_install_record() or _commit_from_git(Path(__file__).resolve().parent)


def _commit_from_install_record() -> str | None:
    try:
        record = metadata.distribution("chessnet").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not record:
        return None
    commit = json.loads(record).get("vcs_info", {}).get("commit_id")
    return commit if isinstance(commit, str) else None


def _commit_from_git(directory: Path) -> str | None:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(directory), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return completed.stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{commit}-dirty" if dirty else commit
