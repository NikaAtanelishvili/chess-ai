# Architecture

This document describes the conventions the code relies on. It grows with the project; it
covers the encoding layer (`chessnet.encoding`), which every other component builds on, and
the network (`chessnet.model`).

## Encoding (`ENCODING_VERSION = 1`)

### Side-to-move perspective

Everything the network sees or predicts is from the point of view of the **side to move**.

- If Black is to move, the board is replaced by `board.mirror()`. python-chess flips the
  board vertically and swaps piece colours, castling rights, the en-passant square and the
  turn. Afterwards the side to move is always "White" and always plays up the board.
- Moves are mirrored the same way: `chess.square_mirror(sq)` (`sq ^ 56`, a vertical flip)
  on both the from- and the to-square. Files do not change, and neither does the promotion
  piece.
- Value targets are win/draw/loss **for the side to move**.

Example: after 1.e4 it is Black's move. The mirrored board shows Black's pawns on rank 2
moving up, and White's e4 pawn as an opponent pawn on e5:

```
real board, Black to move          board.mirror(): what the network sees
8  r n b q k b n r                 8  r n b q k b n r    <- opponent (real White)
7  p p p p p p p p                 7  p p p p . p p p
6  . . . . . . . .                 6  . . . . . . . .
5  . . . . . . . .                 5  . . . . p . . .    <- real White's e4 pawn
4  . . . . P . . .                 4  . . . . . . . .
3  . . . . . . . .                 3  . . . . . . . .
2  P P P P . P P P                 2  P P P P P P P P    <- mover (real Black)
1  R N B Q K B N R                 1  R N B Q K B N R
   a b c d e f g h                    a b c d e f g h
```

Uppercase is White as python-chess prints it. On the right, that is the side to move.

### Tensor orientation

Planes are `planes[plane, rank, file]` with `rank = chess.square_rank(sq)` and
`file = chess.square_file(sq)`. `planes[:, 0, 0]` is a1 from the mover's perspective. The
helper `square_to_rank_file` in `encoding/board.py` owns this mapping, and vectorised code
uses `CELL_SQUARES`, which is derived from it. Since `square == 8 * rank + file`,
flattening a plane row-major yields square order. Batches are NCHW, float32.

### Input planes: `(19, 8, 8)`

| Planes | Content |
|--------|---------|
| 0-5 | Mover's pawns, knights, bishops, rooks, queens, king |
| 6-11 | Opponent's pawns, knights, bishops, rooks, queens, king |
| 12-15 | Constant planes: mover O-O, mover O-O-O, opponent O-O, opponent O-O-O |
| 16 | En-passant target square, one-hot, only if a legal en-passant capture exists |
| 17 | Constant: `min(halfmove_clock, 100) / 100` |
| 18 | All ones (lets convolutions find the board edge under zero padding) |

There are no history or repetition planes. `board.mirror()` drops the move stack, so if
repetition planes are ever added they must be computed before mirroring.

### Policy index: 4672 = 73 move types x 64 from-squares

`index = plane * 64 + from_square`, with `from_square` in the mover's frame. This is the
row-major flattening of a `(73, 8, 8)` convolutional policy head. Directions below are in
the mover's frame (N = towards the opponent).

| Planes | Move type | Order |
|--------|-----------|-------|
| 0-55 | Queen-like moves | `7 * direction + (distance - 1)`; directions N, NE, E, SE, S, SW, W, NW; distance 1-7 |
| 56-63 | Knight moves | (+2,+1), (+1,+2), (-1,+2), (-2,+1), (-2,-1), (-1,-2), (+1,-2), (+2,-1) as (ranks, files), clockwise |
| 64-72 | Underpromotions | `64 + 3 * direction + piece`; directions: capture towards the a-file, straight, capture towards the h-file; pieces: knight, bishop, rook |

- Queen promotions use the queen-like planes (a one-step N, NE or NW pawn move onto the
  last rank). When decoding, a queen-like step of one of the mover's pawns onto the last
  rank becomes a queen promotion.
- Castling is the king's two-square move (`e1g1`, `e1c1`), so it uses the E/W
  distance-2 planes.
- 1858 of the 4672 indices decode to an on-board move; the others raise `ValueError`.
- Examples: `e2e4` = 76, `g1f3` = 4038, `e1g1` = 964, `e1c1` = 2756, `b7a8=N` = 4145.
  Black's mirror-image moves (`e7e5`, `g8f6`, `e8g8`, ...) get the same indices. The unit
  tests pin these values.

API (`encoding/moves.py`, the only implementation of the index):

- `move_to_index(move, board) -> int` mirrors internally based on `board.turn`.
- `index_to_move(index, board) -> chess.Move` returns the real, un-mirrored move, with
  promotions resolved.
- `legal_mask(board) -> np.ndarray[bool] (4672,)`.

### Value target

Three classes from the side to move's perspective: `WIN = 0`, `DRAW = 1`, `LOSS = 2`
(`encoding/value.py`). Search uses the scalar `v = P(win) - P(loss)` in `[-1, 1]`.

### Packed records

Training data is stored as fixed-size records, already in the mover's frame, so training
never calls python-chess (`encoding/packed.py`):

| Field | Type | Content |
|-------|------|---------|
| `bb` | 12 x `<u8` | Bitboards: mover's P, N, B, R, Q, K, then the opponent's |
| `castle` | `u1` | Bits 0-3: mover O-O, mover O-O-O, opponent O-O, opponent O-O-O |
| `ep` | `i1` | En-passant target square if a legal capture exists, else -1 |
| `halfmove` | `u1` | Halfmove clock, clipped to 255 |
| `move` | `<u2` | Policy index of the move played **from** this position |
| `wdl` | `u1` | Outcome for the mover (0 win, 1 draw, 2 loss) |
| `ply` | `<u2` | Half-moves since the game start, from the original board |
| `elo` | `<u2` | Mover's rating, 0 if unknown |

Records are little-endian and unaligned, 106 bytes each. `pack_into` refuses moves that are
not legal in the packed position, so a record can never be labelled with the move that led
to it. `batch_from_records` turns records into int64 tensors on the CPU (bitboards
reinterpreted bit for bit), and `unpack_batch_torch` expands them to `(N, 19, 8, 8)` planes
on the batch's device with vectorised bit operations. The result equals `encode_board`
bit for bit, which the tests check.

Measured on the development machine (4 CPU cores): `encode_board` about 20 µs per
position, `legal_mask` about 31 µs, `pack_into` about 17 µs including the legality check,
and batching plus unpacking about 230k positions/s on the CPU.

### Versioning

Any change to the planes, the move index or the packed format bumps `ENCODING_VERSION`
(`encoding/__init__.py`). That is a breaking change: datasets are regenerated and models
retrained, and old data and checkpoints must fail to load.

## Network (`chessnet.model`)

### Architecture

An SE-ResNet in the style of Leela Chess Zero, with `F` filters and `B` residual blocks:

```
planes (N, 19, 8, 8)
  -> 3x3 conv (19 -> F), BN, ReLU                                    stem
  -> B x [3x3 conv, BN, ReLU, 3x3 conv, BN, SE, + skip, ReLU]        trunk
  -> policy head: 3x3 conv (F -> F), BN, ReLU, 1x1 conv (F -> 73)    -> (N, 73, 8, 8) -> (N, 4672)
  -> value head:  1x1 conv (F -> 32), BN, ReLU, FC 2048 -> 128, ReLU, FC 128 -> 3
```

- **Squeeze-excitation:** global average pooling, then FC `F -> F/4` (ReLU) and
  `F/4 -> 2F`. Half of the output is a per-channel sigmoid scale, the other half a
  per-channel bias. `se_ratio: null` disables it.
- **Policy output:** 4672 logits. The head's `(73, 8, 8)` output is flattened row-major,
  giving `plane * 64 + 8 * rank + file = plane * 64 + square`, exactly the index of
  `encoding/moves.py`. There is no dense layer in between, and a test lights single cells
  for every legal move of a set of positions and checks they land on `move_to_index`.
- **Value output:** win/draw/loss logits for the side to move. `scalar_value` turns them
  into `P(win) - P(loss)` for search.
- Convolutions use Kaiming-normal initialisation; the forward pass returns
  `PolicyValue(policy, value)`, both logits.

### Presets

`configs/model/*.yaml` (and `chessnet.config.MODEL_PRESETS`):

| Preset | Blocks x filters | Parameters | Use |
|--------|------------------|------------|-----|
| `tiny` | 4 x 64 | 626,380 | tests, CI, smoke runs |
| `small` | 6 x 128 | 2,294,092 | cheaper training run |
| `base` | 10 x 128 | 3,526,092 | default Kaggle model |

All three use squeeze-excitation (ratio 4). For `base`, the trunk holds 3,080,000
parameters, the policy head 157,129, the value head 266,819 and the stem 22,144.

### Checkpoints

`model/checkpoint.py` saves one `torch.save` dictionary per checkpoint, written atomically:

- weights; optimizer, scheduler and grad-scaler states; step and epoch;
- Python, NumPy and torch (CPU and CUDA) RNG states;
- the run config, with the model config always under `config["model"]`;
- metrics, and the SHA-256 of the dataset manifest;
- `encoding_version`, the package version, the git commit and a timestamp.

Checkpoints are loaded with `torch.load(..., weights_only=True)`, so opening a downloaded
file cannot run code. **A checkpoint whose `encoding_version` differs from the code's is
refused** (`EncodingVersionError`), as is data built with another version.
`load_network(path)` rebuilds the network from the stored model config.

### Inference and export

- `InferenceModel` puts a network in eval mode and runs it under `torch.inference_mode()`,
  optionally through `torch.compile`.
- `export_onnx` writes a single ONNX file (opset 18, minimum 17) with a dynamic batch
  dimension, for fast CPU play later. Inputs: `planes`; outputs: `policy`, `value`. It
  needs the `onnx` extra (`onnx`, `onnxscript`; `onnxruntime` to run the result). A slow
  test checks that onnxruntime reproduces PyTorch's outputs.
