"""The released network: download it, check it, and put it where the engine looks.

The files come from the GitHub release ``RELEASE_TAG``; their SHA-256 are fixed here, so a
download that differs in any byte is refused.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

from chessnet import __version__

RELEASE_TAG = "v1.0.0"
RELEASE_URL = "https://github.com/NikaAtanelishvili/chess-ai/releases/download/{tag}/{name}"
#: The released files and their SHA-256 (also in the release's SHA256SUMS).
RELEASE_FILES = {
    "base-best.pt": "370248af31086c8de72b9df0ef03a41a426e446a1fe31d6714fd0fe2306deb78",
    "base-best.onnx": "6c02a7d241e5d44d25d962c4e7c8cd5642595badcb7da64c1496ea07788df165",
}
_CHUNK = 1 << 20

Opener = Callable[..., Any]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _open(url: str, timeout: float) -> IO[bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": f"chessnet/{__version__}"})
    response: IO[bytes] = urllib.request.urlopen(request, timeout=timeout)
    return response


def fetch(
    name: str,
    destination: Path,
    *,
    tag: str = RELEASE_TAG,
    expected: str | None = None,
    force: bool = False,
    report: Callable[[str], None] = print,
    opener: Opener = _open,
) -> Path:
    """Download the release file ``name`` to ``destination`` and check its SHA-256.

    A file already at ``destination`` with the right SHA-256 is kept; one with another is
    replaced only with ``force``. The download goes to a temporary file first, so a broken
    or altered transfer never leaves a file in place.
    """
    expected = expected or RELEASE_FILES[name]
    if destination.exists():
        if sha256_of(destination) == expected:
            report(f"{destination}: already there, SHA-256 verified")
            return destination
        if not force:
            raise FileExistsError(
                f"{destination} exists with another SHA-256; pass --force to replace it"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = RELEASE_URL.format(tag=tag, name=name)
    partial = destination.with_name(destination.name + ".part")
    digest, size = hashlib.sha256(), 0
    report(f"downloading {url}")
    try:
        with opener(url, timeout=60) as response, partial.open("wb") as out:
            while chunk := response.read(_CHUNK):
                digest.update(chunk)
                out.write(chunk)
                size += len(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(
                f"{name}: the download's SHA-256 {digest.hexdigest()} is not the released "
                f"{expected}"
            )
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    report(f"{destination}: {size / 1e6:.1f} MB, SHA-256 verified")
    return destination
