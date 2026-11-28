"""The opening book: balanced starting positions, so deterministic engines vary their games.

``data/openings.epd`` holds 50 standard opening lines (6 to 14 plies), each within +-52
centipawns by Stockfish 16 at depth 20 (``ce``, from White's side; White is to move in all).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import chess


@dataclass(frozen=True)
class Opening:
    name: str
    fen: str
    line: str  # the moves from the start position, in SAN

    def board(self) -> chess.Board:
        return chess.Board(self.fen)


def load_openings(path: Path | None = None) -> list[Opening]:
    """The openings in an EPD file (default: the book shipped with the package)."""
    if path is None:
        text = resources.files("chessnet.eval").joinpath("data/openings.epd").read_text()
    else:
        text = path.read_text(encoding="utf-8")
    openings: list[Opening] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        board, operations = chess.Board.from_epd(line)
        name = str(operations.get("id", f"opening {len(openings) + 1}"))
        openings.append(Opening(name, board.fen(), str(operations.get("c0", ""))))
    return openings
