# Legacy code audit

Snapshot audited: tag **`legacy-v0`** → commit `f083561` (identical to `origin/master`
at audit time, 15 commits). All `file:line` references below are against that tag.
Run `scripts/legacy_bug_demo.py` (see [Reproducing](#reproducing-the-findings)) to see
the main defects happen.

**Verdict.** None of the legacy pipeline can be trusted or salvaged. The training
pipeline does not run at the tagged commit. The only existing checkpoint was trained with
an older, incompatible move index. Encoding is inconsistent between training and search,
and the search's backup sign is one ply off, so even a perfect network would play the
worst move it finds. Rebuild from scratch, and keep a few ideas (see [Reusable vs dead](#reusable-vs-dead)).

---

## 1. Inventory

### 1.1 Modules (`src/`)

| File | Purpose | State |
|------|---------|-------|
| `utils.py` | PGN eval filter, board → `(26,8,8)` tensor, board-history builder, PGN iterator | **Runs a file-rewriting filter at import time** (`utils.py:43`); importing it fails today |
| `cnn.py` | ResNet (`in_planes=26`, 128 filters, 10 blocks default) with dense policy head (4672) and tanh value head | Runs |
| `cnn_dataset.py` | `LichessPGNDataset`: whole PGN → in-RAM list of `(board history, move index, eval)` | Crashes (`TypeError`, `cnn_dataset.py:47`) |
| `cnn_train.py` | `Trainer` (Adam, NLL + MSE), plotting, saliency, illegal-move mask helper | `main()` returns before training (`cnn_train.py:125`); `Trainer` would crash (`:57`) |
| `cnn_inference.py` | `CNNInference`: load `state_dict`, run forward pass | Runs; inconsistent return order; `__main__` broken |
| `mcts2.py` | 73-plane move index, canonicalisation helpers, PUCT MCTS (`ChessAI`) with transposition table and eval cache | Runs with a model; sign bug, encoding mismatches |
| `mcts1.py` | Earlier standalone MCTS prototype with a random stub "CNN" | Dead prototype |
| `test.py` | Second copy of the move index plus a 5-case round-trip script (not pytest) | Dead |
| `mean.py`, `chess_board.py`, `start_menu.py` | PyQt5 GUI (start menu, clickable board) | Not connected to the engine: the "bot" plays random moves |
| `requirements.txt` | `pip freeze` dump (CUDA 12.6 torch) | Missing PyQt5 |
| `.gitignore` | Ignores `.idea/`, `archive.py`, `sliding_piece.py` | Legacy |
| `assets/` | Intended piece icons | Empty |

Earlier history (commits before `614ae6e`) also held a hand-written bitboard move generator
(`movegen/`, Cython `sliding_moves/`, `jumping_moves/`, magic-bitboard `.npy/.npz` tables).
It was deleted in `614ae6e` in favour of `python-chess`.

### 1.2 Entry points and how things were launched

There is no package, CLI, or `__init__.py`. Imports use an implicit `src.` namespace package,
so scripts must run from the repo root. The exception is `mean.py`, which mixes `src.chess_board`
and bare `start_menu` imports (`mean.py:6-7`) and only works with both the root and `src/`
on `sys.path` (PyCharm source roots; `.idea/` is present).

| Command | What happens |
|---------|--------------|
| `python -m src.cnn_train` | Builds the dataset from `lichess_test.pgn` in the CWD, prints `dataset[0]`, returns (`cnn_train.py:118-125`); training is commented out (`:143-151`) |
| `python -m src.mcts2` | Loads `chess_model.pth` from the CWD into `CNN(num_blocks=2)`, plays MCTS (2 sims) as White vs a random mover (`mcts2.py:440-460`) |
| `python src/mcts1.py` | Stub MCTS (800 sims, random values) vs random mover |
| `python src/test.py` | Move-index round-trip on 5 hand cases |
| GUI (`mean.py`) | Start menu → board; opponent = `random.choice(legal_moves)` (`chess_board.py:135-139`) |

At the tagged commit, every module that imports `src.utils` (`cnn_dataset`, `cnn_train`,
`mcts2`) fails with `FileNotFoundError: 'lichess_db_standard_rated_2013-01.pgn'`. The cause
is the top-level call at `utils.py:43`, and that PGN has since been deleted from the working tree.

### 1.3 Framework, versions, data format

- **Framework:** PyTorch 2.7.1 (`+cu126` build), python-chess 1.11.2, NumPy 2.2.3,
  Matplotlib 3.10.3, PyQt5 5.15.11. No TensorFlow/Keras anywhere.
- **Raw data:** uncompressed Lichess PGN, January 2013 standard rated games. The filter
  `filter_evaluated_games` (`utils.py:11-41`) reads the whole file into memory and keeps
  games whose every movetext line contains a numeric `[%eval]` comment. Output:
  `lichess_test.pgn`, with 518 games and 34,308 eval comments (31,899 numeric, 2,409 mate scores).
- **Training samples:** python-chess `Board` copies held in RAM (`cnn_dataset.py:19,50`);
  tensors are built per item in `__getitem__`. One-hot 4672 policy target, scalar value target in [-1, 1].
- **Input planes (26):** 6 White pieces, 6 Black pieces, side to move, 4 castling
  rights (absolute colours), halfmove/100, and 4 past positions × 2 occupancy planes.
  There is no en-passant plane.
- **Checkpoints:** bare `state_dict` via `torch.save` (`cnn_train.py:70-71`), with no metadata
  or version.

### 1.4 Where the critical steps happen

| Step | Location |
|------|----------|
| Board encoding (training and search) | `utils.py:45-102` (`board_to_tensor`); the current board is `history[-1]` (`:54`), contradicting the docstring (`:49`) |
| Perspective handling (search only) | `mcts2.py:25-34` (`canonicalize_board`, mirrors by the **AI's colour**, then forces `turn = WHITE` at `:33`); called at `:259` (root), `:333` (leaves), `:432` (history) |
| Move encoding / decoding | `mcts2.py:52-76` (73-entry `ACTIONS` table), `:79-102` (`move_to_index`), `:106-139` (`index_to_move`). Duplicates: commented copy `mcts2.py:140-218`, live copy `test.py:4-91` |
| Policy-target creation | `cnn_dataset.py:47` (index), `:64-66` (one-hot) |
| Value-target creation | `cnn_dataset.py:24,33,40-43` (`[%eval]` in pawns, clamped to ±10, divided by 10) |
| MCTS selection | `mcts2.py:345-363` (`child.Q + c·P·√ΣN/(1+N)`) |
| MCTS backup | `mcts2.py:410-416` (path built at `:318-325`); terminal values `:418-422` |
| Priors | `mcts2.py:277` (root), `:338` (leaves), read at `:351` |

---

## 2. Bug classes

Bug-class labels used in this document:

| # | Bug class | What goes wrong |
|---|-----------|-----------------|
| B1 | Board not mirrored for Black | Input distribution is inconsistent |
| B2 | Board mirrored but move/policy index not mirrored (or the reverse) | The policy target points at the wrong square |
| B3 | Value target not from the side to move's perspective | The value head learns "White wins" instead of "I win" |
| B4 | MCTS backup does not negate the value each ply | Search maximises the opponent's score |
| B5 | Castling/en-passant planes not swapped on mirror | Subtle errors in rights |
| B6 | Promotion handling (queen vs underpromotion, missing promotions) | Illegal or unrepresentable moves |
| B7 | Square→tensor orientation inconsistent between encoder and decoder | Silent garbage |
| B8 | Policy flatten order differs between training and inference | Silent garbage |
| B9 | Softmax/argmax over illegal moves at inference | Illegal moves, or wasted probability mass |
| B10 | Train/validation split by position instead of by game | Inflated validation accuracy (leakage) |
| B11 | Positions from the same game stored contiguously and not shuffled | Correlated batches, value overfitting |
| B12 | Draw/result parsing (`1/2-1/2`, `*`, abandoned games) | Wrong value targets |

### Summary

| # | Status | One-line evidence |
|---|--------|-------------------|
| B1 | **Present** | Training encoder uses absolute colours and never mirrors; search mirrors by AI colour, not side to move |
| B2 | **Present** | Search mirrors the move index by `board.turn` but feeds an unmirrored board; checkpoint was trained with an unmirrored index |
| B3 | **Present** | Value target is the engine eval from White's view; search treats it as side-to-move |
| B4 | **Present** (off-by-one variant) | Negation happens, but starting from the leaf's own perspective; mate-in-1 gets Q = −1 |
| B5 | **Present** (variant) | Castling planes absolute in training, mirrored at search; en passant not encoded at all |
| B6 | **Present** | Decoder turns non-pawn 7th-rank moves into queen promotions; checkpoint-era index could not encode queen promotions |
| B7 | Absent | Encoder and index both use `divmod(sq, 8)` → `(rank, file)`, a1 at `[0, 0]` |
| B8 | Absent | Same `from_sq*73 + action` index in training and search; dense policy head has no spatial coupling |
| B9 | **Present** (wasted mass) | Priors are the raw 4672-way softmax, never masked or renormalised |
| B10 | Absent (no split exists) | No validation or test set at all |
| B11 | Absent | In-RAM dataset with `DataLoader(shuffle=True)` |
| B12 | **Present** (variant) | Result never read; mate-score evals silently dropped; no termination/time-control filter |

### Details

**B1: Present.**
- Training: `board_to_tensor` writes White pieces to planes 0–5 and Black to 6–11 regardless
  of who moves (`utils.py:56-61`) and adds a side-to-move plane (`:64`). The dataset feeds raw
  boards (`cnn_dataset.py:34-35`).
- Search: `canonicalize_board` mirrors when the **AI** is Black (`mcts2.py:27-28`), not when the
  side to move is Black, and then forces `turn = WHITE` (`:33`). At leaves it is applied to
  boards that are already in the canonical frame (`:333`), so for a Black AI the board is mirrored
  twice (back to real orientation). For a White AI, opponent-to-move nodes are fed unmirrored
  Black-to-move boards labelled "White to move".
- Net effect: training sees one encoding and search sees a different one, and neither is a
  consistent side-to-move frame.
- Demo: Black to move after 1.e4 vs its colour mirror → 68 of 1,664 cells differ.

**B2: Present.**
- `move_to_index` mirrors the move when `board.turn == BLACK` (`mcts2.py:81-87`), but the board
  given to the network at that node is not mirrored (see B1). Demo: at the node after 1.e4,
  Black's `e7e5` gets index 877, the same as White's `e2e4`. The network's input has no piece on
  e2 and Black's pawn on e7, so policy and input disagree.
- The existing checkpoint was trained when `move_to_index(move)` had no board argument and never
  mirrored (`git show 614ae6e:src/mcts2.py`, lines 38-57). At the tagged commit it is used with an
  index that mirrors every Black move, so every Black prior is read from the wrong cell.

**B3: Present.**
- The value target is the Lichess `[%eval]` comment: centipawn-style pawns from White's
  perspective, clamped to ±10 and divided by 10 (`cnn_dataset.py:40-43`). It is never flipped to
  the side to move. The search treats the network value as side-to-move (negation in the backup
  at `mcts2.py:416`, checkmate = −1 for the side to move at `:420`).
- Demo: in a game Black won, a Black-to-move sample gets +0.017 (White's view).

**B4: Present (off-by-one variant).**
- `_backpropagate` negates every ply (`mcts2.py:412-416`). But the path holds only child nodes,
  with the leaf first in reverse order (`:318-325`), so the leaf node receives the value from the
  perspective of the side to move *at the leaf*. Its parent then picks `argmax child.Q` (`:358`),
  which maximises the opponent's score.
- Terminal positions make this concrete. A checkmating move leads to a leaf with value −1
  (`:420`), so the parent sees Q = −1 for the mating move and never plays it.
- Demo: with a stub evaluator (uniform priors, value 0) and 300 simulations, mate in 1 is
  visited once, gets Q = −1.00, and is not played, for White and for Black.
- `mcts1.py:172-177` has the same structure (its path includes the root, `:82,88`).

**B5: Present (variant).**
- Castling planes are in absolute colour order in training (`utils.py:68-72`). In search,
  python-chess `mirror()` swaps the rights, so for a Black AI planes 13–14 hold the AI's own
  rights, while in training they always held White's.
- There is no en-passant plane, so en-passant legality is invisible to the network.
- Forcing `canon.turn = WHITE` after mirroring (`mcts2.py:33`) can leave an en-passant square
  that is inconsistent with the side to move.

**B6: Present.**
- Tagged commit: `index_to_move` adds a queen promotion to *any* one-step forward or diagonal
  move from canonical rank 7, whatever the piece (`mcts2.py:126-127`). A rook, bishop, queen or
  king move from the 7th to the 8th rank decodes to an illegal `…q` move. The search pushes
  decoded moves without a legality check (`:323-324`).
- Demo: `a7a8` (rook) → index → `a7a8q`, which is not legal.
- Off-board indices decode to `None` (`:116-117`).
- Checkpoint era (`614ae6e`): queen promotions were unrepresentable (`is_promo = promotion is not
  None` only matched R/B/N), and the underpromotion deltas were `(0,1),(1,1),(-1,1)`, so straight
  promotions were encoded as sideways moves. Black underpromotions failed because nothing was
  mirrored. Those samples were silently skipped by `except ValueError: continue`
  (`cnn_dataset.py:48-49`). The underpromotion action ids also changed meaning between that
  commit and the tag.
- GUI: promotions are impossible because moves are built without a promotion piece
  (`chess_board.py:99,118`).

**B7: Absent.**
- The encoder writes `plane[r, c]` with `r, c = divmod(sq, 8)` (`utils.py:60,90,94`), and the index
  uses the same split (`mcts2.py:89-90,112`): rank-major, a1 at `[0, 0]`.
- Orientation problems in search come from perspective handling (B1/B2) and history ordering
  (X4), not from the square mapping.

**B8: Absent.**
- Index = `from_sq * 73 + action` (`mcts2.py:100`), produced by the same function for targets and
  search, and decoded by its inverse (`:107-109`).
- The policy head is a dense `Linear(73·64 → 4672)` (`cnn.py:34`), so there is no conv-layout
  ordering to get wrong.

**B9: Present (wasted probability mass).**
- Priors are the full 4672-way softmax (`cnn_inference.py:34`), stored as-is (`mcts2.py:277,338`)
  and read per child (`:351`). They are never masked to legal moves or renormalised.
- Children are only created for legal moves (`:368-381`), so illegal moves are not selected
  except through the B6 decode bug.
- The only masking helper, `mask_illegal_moves` (`cnn_train.py:92-97`), is unused and would crash
  (it calls `move_to_index(move)` with one argument).
- Root Dirichlet noise is spread over all 4672 entries, illegal ones included (`mcts2.py:407`).

**B10: Absent, because no split exists.**
- There is no validation or test set anywhere (`cnn_train.py:14-22,118-125`). Nothing leaks, but
  nothing is measured either.

**B11: Absent.**
- The whole dataset lives in RAM as lists of `Board` objects (`cnn_dataset.py:19,50`) and is
  drawn with `DataLoader(shuffle=True)` (`cnn_train.py:18`), so batches are shuffled.
- This doesn't scale: about five `Board` copies per position, and the default `batch_size=1` (`:15`).

**B12: Present (variant).**
- The `Result` header is never read; value targets come from eval comments (B3).
- Mate scores (`[%eval #-3]`) do not match the regex (`cnn_dataset.py:24`) and are silently
  skipped, which drops the most decisive positions (2,409 of 34,308 evals in the bundled sample).
- There is no filter on result (`*`), termination (abandoned), or time control.
  `filter_evaluated_games` only checks for eval comments (`utils.py:23-33`), and bullet games make
  up 213 of the 518 sample games.

### Additional defects found (outside the bug classes above)

- **X1: Policy target is the previous move.** The dataset pushes the move *before* recording the
  position (`cnn_dataset.py:34-35`) and labels it with that same move (`:47,50`). Every sample asks
  the network to predict the move that led to the position, which is never legal in it. Demo:
  input "Black to move after 1.e4", target `e2e4`. This alone makes any training worthless.
- **X2: Training does not run at the tagged commit.** Import-time side effect (`utils.py:43`);
  `move_to_index(move)` missing its `board` argument (`cnn_dataset.py:47`, `cnn_train.py:95`);
  `index_to_move(idx)` missing `turn` (`cnn_train.py:103`); undefined `self.num_samples`
  (`cnn_train.py:57`); `main()` returns before training (`:125`).
- **X3: The only checkpoint is incompatible and unversioned.** It is 89.9 MB with 22.47M
  parameters: 26 input planes, 2 residual blocks, and a 21.8M-parameter dense policy layer. It
  matches `CNN(num_blocks=2)` trained with the `614ae6e` move index (see B2/B6). It has no metadata
  that could have caught the mismatch.
- **X4: History planes are wrong in search.** `_create_canonical_history` appends the current
  position first and older ones after (`mcts2.py:424-438`), but the encoder takes `history[-1]` as
  current (`utils.py:54`). After three moves the root is encoded as the start position (demo).
  Leaf histories shift the root history and ignore the moves along the search path (`mcts2.py:331-334`).
- **X5: Inference API.** `CNNInference.evaluate` returns `(policy, value)` for batch 1 but
  `(value, policy)` otherwise (`cnn_inference.py:39-42`), and prints twice per call in the search
  hot path (`:25,27`). Its `__main__` calls undefined names (`:45-51`).
- **X6: Search details.**
  - The eval cache is keyed by the Zobrist hash of the uncanonicalised board while the input
    depends on history (`mcts2.py:337,387`).
  - First-play urgency is 0 (`Node.Q = 0.0`, `:229`).
  - `is_game_over()` without `claim_draw` misses threefold repetition and the 50-move rule (`:328`).
  - Dirichlet α = 0.03 (`:406`); `mcts1.py:62` applies noise unconditionally.
  - Transposition-table nodes are shared across paths with `parent` overwritten (`:375-377`).
- **X7: Three copies of the move index** (`mcts2.py:52-139`, `:140-218` commented, `test.py:4-91`).
- **X8: Model head.** The dense policy layer is 21.8M of 22.5M parameters (`cnn.py:34`).
  Shape comments are wrong (`cnn.py:48-56`).

---

## 3. Reusable vs dead

**Worth keeping as ideas (not code):**
- The 73-move-type layout (56 queen-like with queen promotions folded in, 8 knight, 9
  underpromotions; `mcts2.py:52-76` at the tagged commit, with the corrected underpromotion deltas).
- `mirror_move` via `chess.square_mirror` (`mcts2.py:16-22`).
- PUCT selection and temperature-based move choice (`mcts2.py:293-303,345-363`).
- Caching evaluations by Zobrist hash.
- The residual block shape (`cnn.py:7-19`).

**Dead:**
- `mcts1.py`, `test.py`, `cnn_inference.py`, `utils.filter_evaluated_games`, `utils.create_board_history`
- the plotting and saliency helpers in `cnn_train.py`
- the PyQt GUI (`mean.py`, `chess_board.py`, `start_menu.py`, empty `assets/`)
- `requirements.txt`
- the existing checkpoint

---

## 4. Datasets and checkpoints on disk

Nothing outside `src/` besides `.venv/` (5.5 GB) and `.git/` (118 MB).

| Path | Size | Where it exists |
|------|------|-----------------|
| `src/lichess_test.pgn` | 1,096,054 B | Working tree; staged (same content as the formerly tracked `…_eval_2013-01.pgn`) |
| `src/chess_model.pth` | 89,912,434 B | **Only in the git index** (staged, then deleted from disk; blob `5d8fd11c6db1…`). It has never been committed or pushed |
| `src/lichess_db_standard_rated_eval_test_1_2013-01.pgn` | 1,822 B | Only in the git index (blob `2091b2842cf7…`) |
| `src/lichess_db_standard_rated_2013-01.pgn` | 92,811,021 B | Committed in history; deleted from the working tree |
| `src/lichess_db_standard_rated_eval_2013-01.pgn` | 1,096,054 B | Committed in history; deleted from the working tree |
| `src/lichess_db_standard_rated_eval_test_2013-01.pgn` | 71,274 B | Committed in history; deleted from the working tree |

The index-only blobs can be recovered with `git cat-file blob <id> > <file>` for as long as
the index keeps them (they are lost if the entries are unstaged and garbage-collected).

---

## 5. Python environment (before the rebuild)

- **Location:** `.venv/` at the repo root, a plain `virtualenv` 20.24.5 venv (not uv-managed),
  based on the system `/usr/bin/python3.12`. Size 5.5 GB.
- **`python --version`:** `Python 3.12.3`
- **Host:** Ubuntu 24.04.3 LTS, 4 CPU cores, 3.6 GiB RAM, ~94 GB free disk, Intel UHD
  integrated graphics only (no NVIDIA GPU; `torch.cuda.is_available()` is `False`).

`pip freeze`:

```
chess==1.11.2
contourpy==1.3.2
cycler==0.12.1
filelock==3.18.0
fonttools==4.58.4
fsspec==2025.5.1
Jinja2==3.1.6
kiwisolver==1.4.8
MarkupSafe==3.0.2
matplotlib==3.10.3
mpmath==1.3.0
networkx==3.5
numpy==2.2.3
nvidia-cublas-cu12==12.6.4.1
nvidia-cuda-cupti-cu12==12.6.80
nvidia-cuda-nvrtc-cu12==12.6.77
nvidia-cuda-runtime-cu12==12.6.77
nvidia-cudnn-cu12==9.5.1.17
nvidia-cufft-cu12==11.3.0.4
nvidia-cufile-cu12==1.11.1.6
nvidia-curand-cu12==10.3.7.77
nvidia-cusolver-cu12==11.7.1.2
nvidia-cusparse-cu12==12.5.4.2
nvidia-cusparselt-cu12==0.6.3
nvidia-nccl-cu12==2.26.2
nvidia-nvjitlink-cu12==12.6.85
nvidia-nvtx-cu12==12.6.77
packaging==25.0
pillow==11.2.1
pyparsing==3.2.3
PyQt5==5.15.11
PyQt5-Qt5==5.15.17
PyQt5_sip==12.17.0
python-dateutil==2.9.0.post0
six==1.17.0
sympy==1.14.0
torch==2.7.1
triton==3.3.1
typing_extensions==4.14.0
```

---

## 6. Repository hygiene

**Secrets: none found.**
- `gitleaks` has **not** been run yet: it is not installed on this machine. Until it is, the
  following substitute scan was done:
  - Every blob of all 15 commits (PGN data excluded) was regex-scanned for AWS, GitHub,
    Slack, OpenAI/Anthropic, Hugging Face and Google API keys, private-key headers and JWTs,
    plus `api_key|secret|password|token|kaggle|wandb` assignments: **0 hits**.
  - No sensitive file names (`.env`, `kaggle.json`, keys, credentials) were ever committed.
  - Working-tree scan: 0 hits. The local `.env` holds a single non-secret username variable;
    it is gitignored and has never been committed.
- To do: run `gitleaks detect` (all history) and `gitleaks detect --no-git` once it is installed.

**Large files in history** (the full history is 118 MB, public):

| Blob | Size | Added → removed |
|------|------|-----------------|
| `src/lichess_db_standard_rated_2013-01.pgn` | 92.8 MB | `614ae6e` → still in `legacy-v0` |
| Cython build output (`*.so`, `*.o`, generated `*.c` under `src/sliding_moves/`, `src/jumping_moves/`) | ~1.0–1.3 MB each, ~10 blobs | `8151cdf`/`0fbba03` → `614ae6e` |
| `src/lichess_db_standard_rated_eval_2013-01.pgn` | 1.1 MB | `614ae6e` → still in `legacy-v0` |
| Magic-bitboard tables (`*.npz`, `*.npy`) | ≤ 63 KB each | deleted in `0fbba03`/`614ae6e` |

The PGN data is Lichess's CC0 open database, so the problem is size only, not licensing or privacy.
No checkpoint has ever been committed.

**Pending local changes at audit time:**
- Pre-staged in the index: 7 `__pycache__/*.pyc` files, `src/lichess_test.pgn` (1.1 MB), and the
  two index-only blobs above.
- Unstaged deletions: the three tracked PGNs.
- `git add -n .` would also have added `.idea/` (including `workspace.xml`) and a `.pyc` file.

None of these belong in git (bytecode, data, checkpoint, IDE state), so none were committed
when freezing. `legacy-v0` tags the last pushed commit, whose code is identical to the
working tree. The index was left untouched so the owner can decide what to do with the
index-only checkpoint. A root `.gitignore` now covers data, checkpoints, logs, environments,
caches, IDE files and `.env`/`kaggle.json`.

---

## Reproducing the findings

```bash
.venv/bin/python scripts/legacy_bug_demo.py
```

It loads the legacy modules from `src/` with the import-time side effect of `utils.py`
removed, writes no files, and prints evidence for B1, B2, B3, B4, B6, B9, X1, X2 and X4.
