"""Streaming PGN sources: local files or HTTP(S) URLs, zstd-compressed or plain text.

Nothing is downloaded to disk or decompressed onto it: bytes flow from the network or the
file through a streaming zstd decompressor into a splitter that yields one game at a time.
"""

from __future__ import annotations

import http.client
import io
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from types import TracebackType
from typing import IO, TYPE_CHECKING, Self

import zstandard

from chessnet import __version__

if TYPE_CHECKING:
    from _typeshed import WriteableBuffer

log = logging.getLogger(__name__)

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
# Skippable frames (magic 0x184D2A50-0x184D2A5F, little-endian) start dumps written by
# parallel compressors such as pzstd, including the Lichess database files.
_SKIPPABLE_FRAME_MAGICS = range(0x184D2A50, 0x184D2A60)
USER_AGENT = f"chessnet/{__version__} (+https://github.com/NikaAtanelishvili/chess-ai)"
GAME_START = b"[Event "
_GAME_SEPARATOR = b"\n" + GAME_START
_RETRYABLE_HTTP_CODES = frozenset({408, 429})


def is_zstd(head: bytes) -> bool:
    """Whether data starting with ``head`` is zstd-compressed (regular or skippable frame)."""
    if len(head) < len(ZSTD_MAGIC):
        return False
    return (
        head[: len(ZSTD_MAGIC)] == ZSTD_MAGIC
        or int.from_bytes(head[:4], "little") in _SKIPPABLE_FRAME_MAGICS
    )


class ResumeNotSupportedError(RuntimeError):
    """The server ignored a ``Range`` request, so a dropped stream cannot be resumed."""


class HTTPStream(io.RawIOBase):
    """Read-only byte stream over an HTTP(S) URL that survives dropped connections.

    After a connection error or a body that ends early, the URL is reopened with
    ``Range: bytes=<offset>-`` and reading continues where it stopped, after an exponential
    backoff (1 s, 2 s, 4 s, ...). Client errors (4xx except 408/429) are not retried.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 60.0,
        max_retries: int = 8,
        backoff: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__()
        self.url = url
        self.position = 0  # bytes received so far
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff = backoff
        self._sleep = sleep
        self._length: int | None = None
        self._response: http.client.HTTPResponse | None = None
        # Own opener: proxy settings are read from the environment when the stream is created.
        self._opener = urllib.request.build_opener()

    def readable(self) -> bool:
        return True

    def _connect(self) -> http.client.HTTPResponse:
        headers = {"User-Agent": USER_AGENT}
        if self.position:
            headers["Range"] = f"bytes={self.position}-"
        request = urllib.request.Request(self.url, headers=headers)
        response: http.client.HTTPResponse = self._opener.open(request, timeout=self._timeout)
        if self.position and response.status != 206:
            response.close()
            raise ResumeNotSupportedError(f"server ignored the Range header for {self.url}")
        if self._length is None:
            length = response.headers.get("Content-Length")
            self._length = int(length) if length else None
        return response

    def readinto(self, buffer: WriteableBuffer) -> int:
        failures = 0
        while True:
            try:
                if self._response is None:
                    self._response = self._connect()
                count = self._response.readinto(buffer)
                if count == 0 and self._length is not None and self.position < self._length:
                    raise http.client.IncompleteRead(b"", self._length - self.position)
                self.position += count
                return count
            except urllib.error.HTTPError as error:
                if error.code < 500 and error.code not in _RETRYABLE_HTTP_CODES:
                    raise
                failures = self._back_off(error, failures)
            except (OSError, http.client.HTTPException) as error:
                failures = self._back_off(error, failures)

    def _back_off(self, error: Exception, failures: int) -> int:
        failures += 1
        if failures > self._max_retries:
            raise error
        if self._response is not None:
            self._response.close()
            self._response = None
        delay = self._backoff ** (failures - 1)
        log.warning(
            "stream interrupted at byte %d (%s); retry %d/%d in %.0f s",
            self.position,
            error,
            failures,
            self._max_retries,
            delay,
        )
        self._sleep(delay)
        return failures

    def close(self) -> None:
        if self._response is not None:
            self._response.close()
            self._response = None
        super().close()


class PGNSource:
    """A PGN dump opened for streaming, with byte counters for progress reports.

    ``source`` is an ``http(s)://`` URL or a local path. Compression is detected from the
    zstd magic number, so ``.pgn.zst`` and plain ``.pgn`` both work.
    """

    def __init__(self, source: str, *, chunk_size: int = 1 << 22) -> None:
        self.source = source
        self.is_url = source.startswith(("http://", "https://"))
        self._chunk_size = chunk_size
        self._http: HTTPStream | None = None
        self._file: io.BufferedReader
        if self.is_url:
            self._http = HTTPStream(source)
            self._file = io.BufferedReader(self._http, buffer_size=1 << 20)
        else:
            self._file = open(source, "rb")  # noqa: SIM115 (closed in close())
        self.compressed = is_zstd(self._file.peek(len(ZSTD_MAGIC)))
        self._stream: IO[bytes] = (
            # A dump is many frames; without read_across_frames the reader stops after the first.
            zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(
                self._file, read_across_frames=True
            )
            if self.compressed
            else self._file
        )
        self.bytes_decompressed = 0

    @property
    def bytes_read(self) -> int:
        """Bytes taken from the network or the file so far (compressed, if compressed)."""
        return self._http.position if self._http is not None else self._file.tell()

    def read(self, size: int) -> bytes:
        data = self._stream.read(size)
        self.bytes_decompressed += len(data)
        return data

    def games(self) -> Iterator[bytes]:
        """The raw text of each game, in file order."""
        return iter_game_blocks(self.read, self._chunk_size)

    def close(self) -> None:
        self._stream.close()
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def iter_game_blocks(
    read: Callable[[int], bytes], chunk_size: int = 1 << 22, *, max_skip: int = 1 << 26
) -> Iterator[bytes]:
    """Split a PGN byte stream into games; a game starts at a line beginning with ``[Event ``.

    Bytes before the first game are skipped (without buffering them); if more than
    ``max_skip`` bytes pass without a game, the input is refused. Each block keeps its
    trailing newlines.

    Raises:
        ValueError: if no game starts within the first ``max_skip`` bytes.
    """
    buffer = b""
    skipped = 0
    found = False
    keep = len(_GAME_SEPARATOR) - 1  # a separator may straddle two chunks
    while True:
        chunk = read(chunk_size)
        buffer += chunk
        if not found:
            if buffer.startswith(GAME_START):
                first = 0
            else:
                index = buffer.find(_GAME_SEPARATOR)
                if index < 0:
                    skipped += max(len(buffer) - keep, 0)
                    buffer = buffer[-keep:]
                    if skipped > max_skip:
                        raise ValueError(
                            f"no PGN game found in the first {skipped:,} bytes; is this a PGN dump?"
                        )
                    if not chunk:
                        return
                    continue
                first = index + 1
            skipped += first
            buffer = buffer[first:]
            found = True
        position = 0
        while (separator := buffer.find(_GAME_SEPARATOR, position)) >= 0:
            yield buffer[position : separator + 1]
            position = separator + 1
        buffer = buffer[position:]
        if not chunk:
            yield buffer
            return
