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
| `wide` | 10 x 192 | 7,572,332 | second Kaggle run, beside `base` |

All use squeeze-excitation (ratio 4). For `base`, the trunk holds 3,080,000
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

## Search (`chessnet.search`) and players (`chessnet.engine`)

### Evaluator

`NetworkEvaluator.evaluate(boards)` returns one `Evaluation(priors, value)` per board:

- It encodes each board from the side to move's perspective and runs the whole batch
  through the network (fp16 autocast on CUDA).
- The **priors** are a softmax over the **legal moves only**: the policy logits are taken at
  the legal moves' indices (`move_to_index`), so no illegal move gets probability.
- The **value** is `P(win) - P(loss)` for the side to move.
- An LRU cache (50,000 positions by default) is keyed by `position_key(board)`: pieces, side
  to move, castling rights, an en-passant square with a legal capture, and the halfmove
  clock up to 100, where its input feature saturates. That is exactly what the network
  sees. History is not part of the key, because the network does not see it.
- `NetworkEvaluator.from_checkpoint(path, device=...)` loads any training checkpoint.

### MCTS (PUCT)

- **Value convention:** values are expected outcomes in [-1, 1]. A node's `value_sum` is
  seen from the player who made the move into the node, so a parent compares its children's
  mean values directly. `backup` adds the leaf value up the path and **negates it at every
  ply**; a test checks the signs on a hand-built tree.
- **Selection:** `argmax Q + c_puct * P * sqrt(N_parent) / (1 + N_child)` with
  `c_puct = 2.0`. An unvisited child scores the parent's own value minus 0.2 (first-play
  urgency) instead of 0, which would over-explore.
- **Terminal positions** are scored without the network: checkmate is -1 for the side to
  move; stalemate, insufficient material, threefold repetition and the 50-move rule are 0.
  Repetitions count the real game: the search pushes its moves onto a copy of the game
  board that keeps the whole move stack (a regression test for the legacy engine's invented
  history). The root itself is always searched, even where a draw could be claimed.
- **Batched leaves with virtual loss:** up to `batch_size` (default 16) leaves share one
  network call. Each leaf selected for the batch counts as a pending loss along its path, so
  the next selections spread out. A leaf selected a second time ends the batch early. The
  first playout evaluates the root alone.
- **Budget:** `search(board, simulations=..., movetime=...)` adds that many playouts, or runs
  for that many seconds, whichever comes first. Each playout adds one visit to the root.
- **Tree reuse:** when the next board continues the same game, the subtree of the moves
  played since is kept, visits included. A board from another game, or the same position
  without its history, starts a new tree.
- **Move choice:** the most visited root move. Players can instead sample the first plies
  from the visit counts with a temperature. There is no Dirichlet noise, which belongs to
  self-play.

### Players

`Player.choose(board) -> Move` is the one interface:

- `RandomPlayer`: a random legal move.
- `PolicyPlayer`: the masked policy's most likely move, with no search.
- `MCTSPlayer(simulations | movetime)`: search with tree reuse between its moves;
  `last_result` keeps the statistics (value, principal variation, nodes, speed).

### Search speed

`scripts/benchmark_search.py` runs 800-playout searches from a fresh tree and an empty cache
in five positions: the start, two openings with Black to move, a middlegame and a rook
endgame. Nodes are playouts. With the base network (`base-best.pt`):

| Hardware | Leaf batch 1 | 8 | 16 | 32 | 64 |
|----------|------:|------:|------:|------:|------:|
| CPU: laptop Intel i3-1115G4 (2 cores, torch 2.15) | 223 | 477 | 537 | 568 | - |
| CPU: Kaggle Xeon 2.0 GHz (2 threads, torch 2.11) | - | 268 | 323 | 367 | - |
| GPU: Kaggle Tesla T4 (fp16, torch 2.11) | 133 | 659 | 986 | 1,305 | 1,507 |

The GPU figures come from the Kaggle kernel `kaggle/search/` (session of 2026-11-26), with
the same checkpoint (its SHA-256 is checked). On the GPU, a lone leaf costs more than on a
CPU, because each call has a fixed latency; batches pay off from 8 leaves up. Batches fill
only partly, because a leaf selected twice ends its batch: 12.8 of 16 and 36 of 64 on
average. The search loop in Python then sets the pace, so a T4 gives about 2-3 times a
laptop CPU at the default batch of 16, not more.

On a CPU the evaluator (encoding plus the network) takes about 95% of the search time.
At batch 16 it evaluates 455 positions per second; the other 15% of playouts are cache
hits and terminal positions, which need no network call. Faster CPU inference, for example
through the ONNX export, would therefore speed up search almost one for one.

## UCI engine (`chessnet.engine.uci`) and the terminal game

### Protocol

`chessnet-uci` (or `chessnet uci`) reads UCI commands from stdin and writes to stdout,
flushing every line:

- `uci` answers at once with the name, the options and `uciok`. Nothing heavy is imported
  until it is needed.
- `isready` loads the network for the current `Weights` and `Device` options (about two
  seconds the first time), then answers `readyok`. During a search it answers at once.
- `position startpos|fen ... moves ...` keeps the moves on the board's stack, so the search
  sees the real game for repetitions, and the next search reuses the tree of the moves
  played. An illegal move is reported with `info string` and the old position is kept.
- `go` starts a search in a worker thread. The main thread keeps reading, so `stop`,
  `isready` and `quit` are handled while the engine thinks. `go infinite` sends its
  `bestmove` only after `stop`, as UCI requires.
- `ucinewgame` drops the tree and the evaluator cache. Unknown commands are ignored.
- A missing or unreadable weights file is reported with `info string`; the engine stays up
  and answers `bestmove 0000`. So does a position with no legal moves.

### Limits and time management

`go nodes N` runs N playouts and `go movetime T` runs T milliseconds, less `MoveOverhead`;
with both, whichever comes first. With a clock (`wtime`/`btime`/`winc`/`binc`), a move
gets `remaining / 30 + 0.8 x increment - overhead` seconds, at most half of the remaining
time after the overhead, and with `movestogo` the time is spread over at most that many
moves. `MoveOverhead` (100 ms) covers the leaf batch that runs past the deadline (about
60 ms on a laptop CPU at batch 32) and the protocol latency. Without node or time limits
(including `go depth`), the engine runs `Simulations` playouts. A test simulates 100-move
games at bullet and blitz controls with a pessimistic cost per move and checks that the
clock never runs out. Another plays real games through python-chess on 5-second and
20-second clocks.

The tree takes about 9 KB per playout, so a search stops growing at half of the memory
available when the engine starts, which allows about 70,000 playouts on a laptop with 1.4 GB
free. An infinite search then waits for `stop` with its result.

### Output and score

During a search, an `info` line appears every second, and a final one before `bestmove`:
`info depth <pv length> score cp <cp> nodes <root visits> nps <playouts per second>
time <ms> pv <moves>`. A move that checkmates is reported as `score mate 1`. The score is
the chosen move's mean value `v` (expected outcome for the engine), converted with
Lichess's logistic win-chance model:

```
v  = 2 / (1 + exp(-0.00368208 * cp)) - 1
cp = ln((1 + v) / (1 - v)) / 0.00368208
```

So `v = 0.18` is about +100 cp and `v = 0.5` about +300 cp. `|v|` is clipped at 0.999, which
keeps scores within about 2,000 cp. The model was fitted to Lichess games, the same
population the network learned from.

### Terminal game

`chessnet play` draws an ASCII board from the human's side and reads moves in SAN or UCI,
plus `moves`, `undo` and `resign`. The engine plays with MCTS (800 playouts by default,
or `--movetime`) or the policy alone, and shows its evaluation for White after each move.
Draws by threefold repetition and the 50-move rule are claimed automatically.
