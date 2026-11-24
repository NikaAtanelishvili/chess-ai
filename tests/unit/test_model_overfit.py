"""The tiny model must be able to memorise a small set of real positions.

If it cannot drive the loss to nearly zero on 64 fixed samples, something between the
encoding, the network and the losses is broken.
"""

import io
from pathlib import Path

import chess
import chess.pgn
import torch
import torch.nn.functional as F  # noqa: N812

from chessnet.config import MODEL_PRESETS
from chessnet.encoding.packed import (
    batch_from_records,
    empty_records,
    pack_into,
    unpack_batch_torch,
)
from chessnet.encoding.value import wdl_from_result
from chessnet.model.resnet import ChessNet

FIXTURE = Path(__file__).parents[1] / "fixtures" / "mini.pgn"
SAMPLES = 64
STEPS = 120


def distinct_samples(count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``count`` distinct positions (ply >= 10) from the fixture games, with their targets."""
    records = empty_records(count)
    seen: set[str] = set()
    filled = 0
    handle = io.StringIO(FIXTURE.read_text(encoding="utf-8"))
    while filled < count and (game := chess.pgn.read_game(handle)) is not None:
        result = game.headers["Result"]
        if result not in ("1-0", "0-1", "1/2-1/2"):
            continue
        board = game.board()
        for move in game.mainline_moves():
            key = f"{board.board_fen()} {board.turn}"
            if board.ply() >= 10 and filled < count and key not in seen:
                seen.add(key)
                pack_into(records, filled, board, move, wdl_from_result(result, board.turn))
                filled += 1
            board.push(move)
    assert filled == count
    batch = batch_from_records(records)
    return unpack_batch_torch(batch), batch.move, batch.wdl


def test_tiny_model_memorises_64_positions() -> None:
    torch.manual_seed(0)
    planes, moves, wdl = distinct_samples(SAMPLES)
    network = ChessNet(MODEL_PRESETS["tiny"])
    optimizer = torch.optim.AdamW(network.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(STEPS):
        output = network(planes)
        loss = F.cross_entropy(output.policy, moves) + F.cross_entropy(output.value, wdl)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < 0.02
    assert bool((output.policy.argmax(dim=1) == moves).all())
    assert bool((output.value.argmax(dim=1) == wdl).all())
