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
- `PolicyPlayer`: the masked policy's most likely move, with no search; the evaluation's
  baseline for the raw network.
- `PolicySafePlayer`: the policy with rules, in prior order. It takes a mate in one. When
  the value head rates the side to move better (`v > 0`), it then plays the first move that
  neither stalemates nor leaves a threefold repetition to claim (completed by the move, or
  reachable by the opponent's reply). Level or worse, or if every move draws, it plays the
  most likely move. UCI `Mode policy` and `chessnet play --mode policy` use it.
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
60 ms on a laptop CPU at batch 32) and the protocol latency. `go depth N` becomes a node
budget of `10 x N^2` playouts, capped at a million, so a GUI's depth setting changes the
strength: depth 1 is 10 playouts, depth 9 is 810 (about the default), depth 20 is 4,000.
With other limits, whichever comes first applies. Without any limit, the engine runs
`Simulations` playouts. A test simulates 100-move
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

## Evaluation (`chessnet.eval`)

### Arena

`run_match` plays N games between two players described by `PlayerSpec`s (random, policy,
MCTS with playouts or time per move, or an external UCI engine such as Stockfish):

- **Openings:** game i starts from opening i // 2 of the book, with the player as White in
  even games and as Black in odd ones. Deterministic players therefore never repeat a game.
  The book (`chessnet/eval/data/openings.epd`) has 50 standard lines of 6 to 14 plies, each
  within +-52 cp by Stockfish 16 at depth 20.
- **Adjudication:** checkmate, stalemate, insufficient material, threefold repetition that
  has happened (a merely available repetition does not end the game, so a winning side
  is not drawn early), the 50-move rule, or 600 plies (a draw). An illegal move loses.
- **Statistics:** wins, draws and losses from the player's side; score; the Elo
  difference `-400 log10(1/score - 1)`. The 95% interval comes from the per-game variance
  of the score, or from the Wilson interval when every game had the same result, so 100
  wins in 100 games still give a finite lower bound (about +550).
- **Parallel games:** with `--workers N`, N worker processes each build the players once
  (networks, search trees, engine processes) and take games as they free up, under the
  supervisor described below. Torch threads are shared out, and GPUs are assigned
  round-robin.
- **PGNs:** every game of a match, with the opening's FEN, the players, the result and the
  termination.

### Puzzles

Lichess puzzle rows give a FEN, the moves and a rating. The player sees the position after
the first move (the opponent's) and must find the solution's first move. Any checkmate
counts when the solution checkmates, as on Lichess. Accuracy is reported for four rating
buckets: up to 1000, 1001-1500, 1501-2000, and above 2000. Samples take equal numbers per
bucket, uniformly among established puzzles (rating deviation at most 100, popularity at
least 80, at least 200 plays), and are reproducible for a given file and seed. The
committed fixture has 125 per bucket from the November 2026 database.

### Suites

| Suite | Matches | Puzzles |
|-------|---------|---------|
| quick | policy vs random (20 games); policy-safe vs random (20); MCTS 400 vs policy (20) | 500 (fixture): policy, MCTS 200 |
| full | policy vs random (100); policy-safe vs random (100); MCTS 400 vs policy (100); MCTS 800 vs Stockfish at its lowest `UCI_Elo` (50) and at `Skill Level` 0 (50), 1 s per Stockfish move; with `--compare`, the second network vs base at 800 playouts each (50) and at 1 s per move each (50) | 2,000 sampled from the database: policy, MCTS 800 |

`--estimate` plays two games of each match and ten puzzles of each run, and extrapolates.
`--only <part>,...` runs just those matches or puzzle runs (`puzzles-policy`,
`puzzles-mcts-800`) and merges them into the suite's existing results (for the same
checkpoint). Each part records the hardware, date and commit it ran on, and the model card
says so when the hardware differs from the rest of the suite. The results list the parts
of the suite that have not run yet (`not_run`). With `--games N --label TEXT` the chosen
matches are played as labelled variants, kept next to the planned ones: the laptop's
20-game equal-time comparison is `--only compare-equal-time --games 20 --label "laptop CPU"
--device cpu`. `--merge-results <file>` merges a results file of the same suite and
checkpoint from another machine, with its PGNs, into ours.

Matches in which a search has a time limit per move (the equal-time comparison) play one
game per GPU, or one at a time on a CPU, whatever `--workers` says: parallel games would
take each other's compute, and the time limit would buy fewer playouts.

### Stockfish ladder and playout scaling

`chessnet ladder` (`chessnet.eval.ladder`) rates MCTS on Stockfish's scale:

- **Ladder:** MCTS 800 against Stockfish 16 with `UCI_LimitStrength` at UCI_Elo 1500, 1700,
  1900, 2100 and 2300 (40 games each, 1 s per Stockfish move, the balanced openings with
  both colours). It stops climbing after the first level scored below 25%.
- **Rating:** the maximum-likelihood fit of the Elo model to every ladder game (a draw is
  half a win), with a 95% likelihood-ratio interval. Stockfish 16 turns UCI_Elo into a
  fractional skill level (1320 is level 0) fitted to CCRL engine ratings, so the result is
  an engine rating on that scale, not a FIDE or Lichess rating.
- **Scaling:** MCTS at 50, 100, 200, 400, 800 and 1,600 playouts (32 games each) against the
  ladder level where MCTS 800 scored closest to 50%, where an Elo difference is measured
  most precisely. A point's rating is that level plus its Elo difference; the 800 point is
  the ladder's own match. A straight line in log2(playouts), fitted to every scaling game
  by maximum likelihood, gives the Elo per doubling with a profile-likelihood interval; a
  likelihood-ratio test of a curvature term says whether the curve flattens. The chart
  (`<checkpoint>.ladder.svg`: the points with 95% intervals and the fitted line, light and
  dark) and the tables go to the model card.
- The same supervised workers, heartbeat, per-match saving and `--max-wall-time` as the
  suites. `--import-results` brings a run from another machine, with its PGNs; with
  `--merge` its levels join an earlier ladder of the same playouts (a ladder extended
  upwards). `--name ladder-1600` keeps a second ladder, for another playout budget, in its
  own results file and model card section.
TensorBoard scalars go to `eval/results/tb/`.

### Supervised runs

`chessnet eval` plays every game and puzzle in worker processes under a supervisor
(`chessnet.eval.workers`), so that no engine, network or deadlock can stall a run:

- **Limits:** a worker that plays no move for 5 minutes, or spends more than an hour on one
  game or puzzle, is killed together with its engines (each worker leads its own process
  group) and replaced. An external engine also has a limit per call: python-chess gives up
  after the move time plus 30 s (`PlayerSpec.timeout`), and the worker restarts its
  players. Either way the game is recorded as an error, with its reason, and left out of
  the score; the model card says how many games did not finish.
- **No endless retries:** a worker that cannot build its players hands its game back.
  After 4 workers in a row die without finishing anything, the rest of the match fails
  with the reason.
- **Heartbeat:** every 60 s, `[match] heartbeat: 37/50 done, 4 running, 12.3 min`.
- **Saving:** each match is written (JSON and PGN) as soon as it ends, so a crash or a
  cancelled session loses at most the match in progress.
- **Wall time:** `--max-wall-time 6h` stops the suite cleanly. The running match or puzzle
  run is cut off and saved as incomplete, and the parts that never started are listed as
  not run.
- **Exit:** workers close their players themselves before exiting. python-chess drives each
  engine from a non-daemon thread, and Python waits for those threads before it runs exit
  handlers, so a worker that left an engine open would never exit.
