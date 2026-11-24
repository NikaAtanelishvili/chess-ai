"""Compact training records and their unpacking into input planes.

A record stores one training example (position, move played from it, outcome) *after*
the perspective transform, so reading records never needs python-chess:

============ ======= ===========================================================
field        type    content
============ ======= ===========================================================
``bb``       12 x u8 bitboards: mover's then opponent's P, N, B, R, Q, K
``castle``   u1      bits 0-3: mover O-O, mover O-O-O, opponent O-O, opponent O-O-O
``ep``       i1      en-passant target square if a legal capture exists, else -1
``halfmove`` u1      halfmove clock, clipped to 255
``move``     u2      policy index of the move played (see ``encoding.moves``)
``wdl``      u1      outcome for the mover: 0 win, 1 draw, 2 loss
``ply``      u2      half-moves since the game start (original board), clipped
``elo``      u2      mover's rating, 0 if unknown
============ ======= ===========================================================

Records are little-endian and unaligned: 106 bytes each.

Packing needs NumPy only. The PyTorch side (:class:`PackedBatch`,
:func:`unpack_batch_torch`) imports torch lazily so data-preparation workers stay light.
Unpacking runs on whatever device the batch is on, typically the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any, Final

import chess
import numpy as np
import numpy.typing as npt

from chessnet.encoding.board import (
    CASTLING_PLANE,
    CELL_SQUARES,
    EN_PASSANT_PLANE,
    HALFMOVE_CLOCK_MAX,
    HALFMOVE_FEATURE,
    HALFMOVE_PLANE,
    NUM_PLANES,
    ONES_PLANE,
    OPPONENT_PIECE_PLANE,
    PIECE_TYPES,
    castling_flags,
    en_passant_square,
    perspective_board,
)
from chessnet.encoding.moves import move_to_index
from chessnet.encoding.value import WDL_CLASSES

if TYPE_CHECKING:
    import torch

PACKED_DTYPE: Final = np.dtype(
    [
        ("bb", "<u8", (12,)),
        ("castle", "u1"),
        ("ep", "i1"),
        ("halfmove", "u1"),
        ("move", "<u2"),
        ("wdl", "u1"),
        ("ply", "<u2"),
        ("elo", "<u2"),
    ]
)
RECORD_SIZE: Final = PACKED_DTYPE.itemsize  # 106 bytes
MAX_PLY: Final = 65535
MAX_ELO: Final = 65535

PackedRecords = npt.NDArray[np.void]


def empty_records(count: int) -> PackedRecords:
    """A zero-filled array of ``count`` packed records."""
    return np.zeros(count, dtype=PACKED_DTYPE)


def pack_into(
    records: PackedRecords,
    index: int,
    board: chess.Board,
    move: chess.Move,
    wdl: int,
    *,
    elo: int = 0,
) -> None:
    """Write the example "``move`` was played from ``board``" into ``records[index]``.

    ``wdl`` is the game's outcome for the side to move in ``board``
    (see :func:`chessnet.encoding.value.wdl_from_result`).

    Raises:
        ValueError: if ``move`` is not legal in ``board`` (a training target must be a
            move played *from* the encoded position), or ``wdl`` or ``elo`` is invalid.
    """
    if wdl not in WDL_CLASSES:
        raise ValueError(f"wdl must be one of {WDL_CLASSES}, got {wdl}")
    if not 0 <= elo <= MAX_ELO:
        raise ValueError(f"elo must be in [0, {MAX_ELO}], got {elo}")
    if not board.is_legal(move):
        raise ValueError(f"move {move.uci()} is not legal in {board.fen()}")

    view = perspective_board(board)
    records["bb"][index] = [
        view.pieces_mask(piece_type, color)
        for color in (chess.WHITE, chess.BLACK)
        for piece_type in PIECE_TYPES
    ]
    records["castle"][index] = sum(
        1 << bit for bit, has_right in enumerate(castling_flags(view)) if has_right
    )
    ep_square = en_passant_square(view)
    records["ep"][index] = -1 if ep_square is None else ep_square
    records["halfmove"][index] = min(board.halfmove_clock, HALFMOVE_CLOCK_MAX)
    records["move"][index] = move_to_index(move, board)
    records["wdl"][index] = wdl
    records["ply"][index] = min(board.ply(), MAX_PLY)
    records["elo"][index] = elo


def pack_position(board: chess.Board, move: chess.Move, wdl: int, *, elo: int = 0) -> PackedRecords:
    """A single packed record (array of shape ``(1,)``); see :func:`pack_into`."""
    records = empty_records(1)
    pack_into(records, 0, board, move, wdl, elo=elo)
    return records


def unpack_board(record: np.void) -> chess.Board:
    """The position of a record as the mover saw it (White to move; see ``perspective_board``).

    Only what a record stores is restored: pieces, castling rights, a legal en-passant
    square, the halfmove clock (clipped at 255) and the move number derived from ``ply``.
    """
    board = chess.Board.empty()
    for index, bitboard in enumerate(record["bb"].tolist()):
        color = chess.WHITE if index < len(PIECE_TYPES) else chess.BLACK
        piece = chess.Piece(PIECE_TYPES[index % len(PIECE_TYPES)], color)
        for square in chess.SquareSet(bitboard):
            board.set_piece_at(square, piece)
    castle = int(record["castle"])
    rights = "".join(symbol for bit, symbol in enumerate("KQkq") if castle >> bit & 1)
    board.set_castling_fen(rights or "-")
    ep_square = int(record["ep"])
    board.ep_square = ep_square if ep_square >= 0 else None
    board.halfmove_clock = int(record["halfmove"])
    board.fullmove_number = int(record["ply"]) // 2 + 1
    return board


@dataclass(frozen=True, slots=True)
class PackedBatch:
    """Packed records as int64 tensors of shape ``(N, 12)`` (bitboards) or ``(N,)``.

    ``bitboards`` holds the uint64 bitboards reinterpreted bit for bit as int64.
    """

    bitboards: torch.Tensor
    castling: torch.Tensor
    en_passant: torch.Tensor
    halfmove: torch.Tensor
    move: torch.Tensor
    wdl: torch.Tensor
    ply: torch.Tensor
    elo: torch.Tensor

    def __len__(self) -> int:
        return int(self.bitboards.shape[0])

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> PackedBatch:
        """Copy every tensor to ``device``."""
        moved: dict[str, Any] = {
            field.name: getattr(self, field.name).to(device, non_blocking=non_blocking)
            for field in fields(self)
        }
        return replace(self, **moved)


def batch_from_records(records: PackedRecords) -> PackedBatch:
    """Convert packed records (e.g. a slice of a memory-mapped shard) to CPU tensors."""
    import torch

    # Copy into freshly allocated arrays. Shortcuts such as np.ascontiguousarray may return
    # the strided field view itself (e.g. for a single record), which torch rejects.
    count = len(records)

    def column(name: str) -> torch.Tensor:
        values = np.empty(count, dtype=np.int64)
        values[:] = records[name]
        return torch.from_numpy(values)

    bitboards = np.empty((count, 12), dtype="<u8")
    bitboards[:] = records["bb"]
    return PackedBatch(
        bitboards=torch.from_numpy(bitboards.view(np.int64)),
        castling=column("castle"),
        en_passant=column("ep"),
        halfmove=column("halfmove"),
        move=column("move"),
        wdl=column("wdl"),
        ply=column("ply"),
        elo=column("elo"),
    )


def unpack_batch_torch(batch: PackedBatch) -> torch.Tensor:
    """Input planes ``(N, 19, 8, 8)`` float32 for a batch, on the batch's device.

    Equal, bit for bit, to :func:`chessnet.encoding.board.encode_board` of each position.
    """
    import torch

    device = batch.bitboards.device
    count = len(batch)
    # Cell k of a flattened plane shows square CELL_SQUARES[k]; shifting by it reads that bit.
    # Arithmetic right shifts of the int64 view still yield the right bit after "& 1".
    cells = torch.tensor(CELL_SQUARES, dtype=torch.int64, device=device)
    castling_bits = torch.arange(4, dtype=torch.int64, device=device)
    halfmove_feature = torch.tensor(HALFMOVE_FEATURE, device=device)

    planes = torch.empty((count, NUM_PLANES, 64), dtype=torch.float32, device=device)
    planes[:, : OPPONENT_PIECE_PLANE + len(PIECE_TYPES)] = (
        batch.bitboards.unsqueeze(-1) >> cells
    ) & 1
    planes[:, CASTLING_PLANE : CASTLING_PLANE + 4] = (
        (batch.castling.unsqueeze(-1) >> castling_bits) & 1
    ).unsqueeze(-1)
    planes[:, EN_PASSANT_PLANE] = batch.en_passant.unsqueeze(-1) == cells
    planes[:, HALFMOVE_PLANE] = halfmove_feature[batch.halfmove].unsqueeze(-1)
    planes[:, ONES_PLANE] = 1.0
    return planes.view(count, NUM_PLANES, 8, 8)
