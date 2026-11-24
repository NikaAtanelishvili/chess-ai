import io
import threading
import urllib.error
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import zstandard

from chessnet.data.pgn_stream import (
    HTTPStream,
    PGNSource,
    ResumeNotSupportedError,
    is_zstd,
    iter_game_blocks,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"
MINI = (FIXTURES / "mini.pgn").read_bytes()
MINI_ZST = (FIXTURES / "mini.pgn.zst").read_bytes()
TWO_GAMES = b'[Event "a"]\n\n1. e4 1-0\n\n[Event "b"]\n\n1. d4 0-1\n'


def reader(data: bytes) -> Callable[[int], bytes]:
    return io.BytesIO(data).read


def test_compressed_fixture_matches_plain_fixture() -> None:
    assert zstandard.ZstdDecompressor().decompress(MINI_ZST) == MINI


def test_games_are_split_at_event_lines() -> None:
    assert list(iter_game_blocks(reader(TWO_GAMES))) == [
        b'[Event "a"]\n\n1. e4 1-0\n\n',
        b'[Event "b"]\n\n1. d4 0-1\n',
    ]


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 64])
def test_tiny_chunks_split_the_same_way(chunk_size: int) -> None:
    blocks = list(iter_game_blocks(reader(TWO_GAMES), chunk_size))
    assert blocks == list(iter_game_blocks(reader(TWO_GAMES)))


@pytest.mark.parametrize("chunk_size", [97, 1000, 4096, 1 << 22])
def test_fixture_splits_into_twenty_games(chunk_size: int) -> None:
    blocks = list(iter_game_blocks(reader(MINI), chunk_size))
    assert len(blocks) == 20
    assert all(block.startswith(b"[Event ") for block in blocks)
    assert b"".join(blocks) == MINI


def test_bytes_before_the_first_game_are_skipped() -> None:
    data = b"\xef\xbb\xbfsome preamble\nmore\n" + TWO_GAMES
    assert list(iter_game_blocks(reader(data), 5)) == list(iter_game_blocks(reader(TWO_GAMES)))


@pytest.mark.parametrize("data", [b"", b"no games here\n", b"[Event without a newline before"])
def test_input_without_games(data: bytes) -> None:
    blocks = list(iter_game_blocks(reader(data)))
    assert blocks == ([data] if data.startswith(b"[Event ") else [])


def test_crlf_line_endings() -> None:
    data = TWO_GAMES.replace(b"\n", b"\r\n")
    assert len(list(iter_game_blocks(reader(data)))) == 2


@pytest.mark.parametrize(("name", "compressed"), [("mini.pgn", False), ("mini.pgn.zst", True)])
def test_local_sources(name: str, compressed: bool) -> None:
    path = FIXTURES / name
    with PGNSource(str(path)) as source:
        assert not source.is_url
        assert source.compressed is compressed
        games = list(source.games())
        assert source.bytes_read == path.stat().st_size
        assert source.bytes_decompressed == len(MINI)
    assert len(games) == 20


class _Server(ThreadingHTTPServer):
    """Serves one payload; supports "Range: bytes=N-" and a few failure modes."""

    def __init__(
        self,
        payload: bytes,
        *,
        drop_after: int | None = None,
        ignore_range: bool = False,
        status: int = 200,
    ) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.payload = payload
        self.drop_after = drop_after
        self.ignore_range = ignore_range
        self.status = status
        self.ranges: list[str | None] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/dump.pgn.zst"


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, format: str, *args: object) -> None:
        pass  # keep test output quiet

    def do_GET(self) -> None:
        server = self.server
        requested = self.headers.get("Range")
        server.ranges.append(requested)
        if server.status != 200:
            self.send_error(server.status)
            return
        start = 0
        if requested and not server.ignore_range:
            start = int(requested.removeprefix("bytes=").rstrip("-"))
            self.send_response(206)
            total = len(server.payload)
            self.send_header("Content-Range", f"bytes {start}-{total - 1}/{total}")
        else:
            self.send_response(200)
        body = server.payload[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if server.drop_after is not None and len(server.ranges) == 1:
            self.wfile.write(body[: server.drop_after])  # then hang up early
            self.close_connection = True
            return
        self.wfile.write(body)


@contextmanager
def serve(payload: bytes, **options: object) -> Iterator[_Server]:
    server = _Server(payload, **options)  # type: ignore[arg-type]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(autouse=True)
def _no_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)


def test_url_source_streams_all_games() -> None:
    with serve(MINI_ZST) as server, PGNSource(server.url) as source:
        assert source.is_url
        assert source.compressed
        assert len(list(source.games())) == 20
        assert source.bytes_read == len(MINI_ZST)
    assert server.ranges == [None]


def test_dropped_connection_resumes_with_a_range_request() -> None:
    delays: list[float] = []
    with serve(MINI_ZST, drop_after=1000) as server:
        stream = HTTPStream(server.url, sleep=delays.append)
        assert io.BufferedReader(stream).read() == MINI_ZST
    assert server.ranges == [None, "bytes=1000-"]
    assert delays == [1.0]


def test_resume_needs_range_support() -> None:
    with serve(MINI_ZST, drop_after=1000, ignore_range=True) as server:
        stream = HTTPStream(server.url, sleep=lambda _: None)
        with pytest.raises(ResumeNotSupportedError):
            io.BufferedReader(stream).read()


def test_client_errors_are_not_retried() -> None:
    delays: list[float] = []
    with serve(MINI_ZST, status=404) as server:
        with pytest.raises(urllib.error.HTTPError, match="404"):
            HTTPStream(server.url, sleep=delays.append).read(10)
        assert server.ranges == [None]
    assert delays == []


def test_server_errors_are_retried_with_backoff_then_raised() -> None:
    delays: list[float] = []
    with serve(MINI_ZST, status=503) as server:
        with pytest.raises(urllib.error.HTTPError, match="503"):
            HTTPStream(server.url, max_retries=3, sleep=delays.append).read(10)
        assert len(server.ranges) == 4
    assert delays == [1.0, 2.0, 4.0]


def test_unreachable_server_is_retried_then_raised() -> None:
    with serve(b"") as server:
        url = server.url
    delays: list[float] = []
    with pytest.raises(urllib.error.URLError):
        HTTPStream(url, max_retries=1, sleep=delays.append).read(10)
    assert delays == [1.0]


def skippable_frame(payload: bytes) -> bytes:
    """A zstd skippable frame (as written by parallel compressors such as pzstd)."""
    return (0x184D2A50).to_bytes(4, "little") + len(payload).to_bytes(4, "little") + payload


def multi_frame(data: bytes, frames: int) -> bytes:
    """``data`` compressed as several independent zstd frames, each after a skippable frame."""
    size = -(-len(data) // frames)
    compressor = zstandard.ZstdCompressor()
    return b"".join(
        skippable_frame(b"\x00\x01\x02\x03") + compressor.compress(data[start : start + size])
        for start in range(0, len(data), size)
    )


@pytest.mark.parametrize("frames", [1, 2, 7])
def test_multi_frame_dumps_with_skippable_frames(tmp_path: Path, frames: int) -> None:
    # Lichess dumps start with a skippable frame and contain many frames.
    path = tmp_path / "dump.pgn.zst"
    path.write_bytes(multi_frame(MINI, frames))
    with PGNSource(str(path)) as source:
        assert source.compressed
        games = list(source.games())
        assert source.bytes_decompressed == len(MINI)
    assert len(games) == 20


def test_long_junk_prefix_is_skipped_with_bounded_buffering() -> None:
    junk = bytes(range(256)) * 4096  # 1 MiB without any game start
    blocks = list(iter_game_blocks(reader(junk + b"\n" + TWO_GAMES), 4096))
    assert blocks == list(iter_game_blocks(reader(TWO_GAMES)))


def test_input_that_is_not_pgn_is_refused() -> None:
    junk = bytes(range(256)) * 4096
    with pytest.raises(ValueError, match="no PGN game"):
        list(iter_game_blocks(reader(junk), 4096, max_skip=100_000))


@pytest.mark.parametrize(
    ("head", "expected"),
    [
        (b"\x28\xb5\x2f\xfd", True),
        (b"\x50\x2a\x4d\x18", True),  # skippable frame
        (b"\x5f\x2a\x4d\x18", True),  # last skippable-frame magic
        (b"\x60\x2a\x4d\x18", False),
        (b"[Eve", False),
        (b"\x28\xb5", False),  # too short to tell
        (b"", False),
    ],
)
def test_zstd_detection(head: bytes, expected: bool) -> None:
    assert is_zstd(head) is expected


def test_empty_source(tmp_path: Path) -> None:
    path = tmp_path / "empty.pgn"
    path.write_bytes(b"")
    with PGNSource(str(path)) as source:
        assert not source.compressed
        assert list(source.games()) == []
