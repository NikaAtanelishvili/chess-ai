"""Where the running code came from, for dataset manifests and checkpoints."""

from __future__ import annotations

import json
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


def git_commit() -> str | None:
    """Commit of the running chessnet code, or ``None`` if it cannot be determined.

    Sources, in order:

    1. the installer's record of a VCS install (``pip install git+...@<ref>``);
    2. git in the directory a local install came from (``pip install /path/to/checkout``,
       as the Kaggle kernels do);
    3. git in the package's own source tree (editable installs, development).

    ``-dirty`` is appended when git reports uncommitted changes to tracked files.
    """
    record = _install_record()
    commit = record.get("vcs_info", {}).get("commit_id")
    if isinstance(commit, str):
        return commit
    url = record.get("url")
    if isinstance(url, str) and url.startswith("file://"):
        source = Path(unquote(urlparse(url).path))
        commit = _commit_from_git(source)
        if commit is not None:
            return commit
    return _commit_from_git(Path(__file__).resolve().parent)


def _install_record() -> dict[str, Any]:
    """The installer's ``direct_url.json`` for chessnet (PEP 610), or an empty dict."""
    try:
        text = metadata.distribution("chessnet").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return {}
    record = json.loads(text) if text else {}
    return record if isinstance(record, dict) else {}


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
