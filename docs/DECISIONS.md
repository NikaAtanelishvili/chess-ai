# Decision log

Short records of non-obvious choices, newest last. Each entry: context, decision,
consequences.

## 2026-11-24: Freeze the legacy engine as `legacy-v0` and rebuild

**Context.** The audit (`docs/AUDIT.md`) found the training pipeline broken, the only
checkpoint incompatible with the code, and sign and perspective bugs in search.
**Decision.** Tag the last pushed commit as `legacy-v0` and rebuild from scratch. The
pending local changes (bytecode, PGN data, an invalid checkpoint) were not committed, and
the checkpoint was discarded. Published history is never rewritten; the PGN dumps in it
are CC0 data and contain no secrets.
**Consequences.** `legacy-v0` is the only place the old code and data live once `legacy/`
is deleted.

## 2026-11-24: Keep legacy code in `legacy/` with its imports untouched

**Context.** The new package needs `src/`, and the legacy modules import each other as
`src.*`.
**Decision.** Move the legacy modules unchanged to `legacy/`, exclude them from lint,
type checks, coverage and pre-commit, and let `scripts/legacy_bug_demo.py` alias `src`
to `legacy/` at runtime. The PyQt GUI was deleted instead of moved (owner decision: play
through UCI GUIs and a terminal command).
**Consequences.** The legacy code is not importable as a normal package; only the demo
script loads it.

## 2026-11-24: Build backend and versioning

**Decision.** Hatchling, with the version read from `chessnet.__version__`, so there is
one place to bump it.

## 2026-11-24: PyTorch from the CPU wheel index, lower bound only

**Context.** The development machine has no NVIDIA GPU (4 cores, 3.6 GiB RAM). Training
runs on Kaggle, where a CUDA build of torch is preinstalled.
**Decision.** `torch>=2.3` in the package metadata, and uv resolves it from the PyTorch
CPU index (`[[tool.uv.index]]` with `explicit = true` plus `[tool.uv.sources]`). Exact
pins live in `uv.lock`.
**Consequences.** Local and CI installs avoid multi-GB CUDA wheels. `pip install` on Kaggle
keeps the preinstalled CUDA torch because the metadata asks for no specific build.

## 2026-11-24: Python 3.11 for development, 3.11 and 3.12 in CI

**Decision.** `.python-version` pins 3.11 (close to Kaggle's images); `requires-python =
">=3.11"`; CI tests both 3.11 and 3.12.
**Follow-up.** When the Kaggle training kernel is set up, check the Python version of
Kaggle's current GPU image. If it is newer than the pin, align `.python-version` (and add
that version to the CI matrix).

## 2026-11-24: Dependencies and extras

**Decision.** The runtime dependencies every planned command needs (python-chess, NumPy,
PyTorch, typer, pydantic, PyYAML, zstandard) are declared now, so CI exercises the real
install from the start. Developer tools live in the PEP 735 `dev` group. The `train` extra
holds TensorBoard. An `eval` extra and an opt-in W&B extra will be added when they have
contents; Stockfish is an external binary, not a Python dependency.

## 2026-11-24: Per-package strict typing

**Context.** mypy's `strict` option cannot be set per module.
**Decision.** A moderate global baseline, plus an override listing the individual strict
flags for `chessnet.encoding`, `chessnet.search` and `chessnet.model`.

## 2026-11-24: Secret scanning

**Context.** The repository is public. The gitleaks GitHub Action only scans the new commit
range on push and pull request events, and the full history only on scheduled and manual
runs.
**Decision.** The `Secret scan` workflow runs the gitleaks Action (incremental, with PR
comments; full history weekly and on demand) and, on every push and pull request, a
full-history scan with the pinned `ghcr.io/gitleaks/gitleaks` image. The pre-commit hook
builds gitleaks from source, so nothing has to be installed locally.

## 2026-11-24: PR title check on `pull_request`

**Decision.** `amannn/action-semantic-pull-request` runs on `pull_request` rather than
`pull_request_target`. It only needs a read token for same-repository PRs, and it also
runs on the PR that introduces or changes it.

## 2026-11-24: Data and resource constraints (owner decisions)

**Decision.**
- The training data's rating floor is 2000 for both players.
- Preprocessing runs in a Kaggle CPU kernel, not locally.
- Local tests and evaluation stay memory-light: the tiny model in tests, memory-mapped
  shards, and no loading of whole datasets.
- lichess-bot is out of scope for now.

## 2026-11-24: Move-index layout details (encoding version 1)

**Context.** The 8x8x73 layout fixes the move-type groups but not their internal order.
**Decision.**
- Queen-like directions run clockwise from N (towards the opponent): N, NE, E, SE, S,
  SW, W, NW, each with distances 1-7, so `plane = 7 * direction + distance - 1`.
- Knight jumps run clockwise from (+2 ranks, +1 file).
- Underpromotions are direction-major: `64 + 3 * direction + piece`, with directions
  (towards the a-file, straight, towards the h-file) and pieces (knight, bishop, rook).
- `index_to_move` turns a queen-like step of the mover's pawn onto the last rank into a
  queen promotion. Underpromotion planes are valid only from the 7th rank. 1858 indices
  decode to on-board moves; the rest raise `ValueError`.

Golden indices in the unit tests pin all of this.

## 2026-11-24: En passant only when a capture is legal

**Context.** python-chess keeps a potential en-passant square after every double pawn push,
even when no pawn can capture.
**Decision.** Plane 16 (and the packed `ep` field) is set only if
`board.has_legal_en_passant()`. That is the only case where it matters for move choice,
and it matches python-chess's repetition key, so positions that are the same for
repetition purposes encode identically.

## 2026-11-24: Halfmove clock storage

**Decision.** Records store the clock clipped to 255 (one byte). The plane value
`min(clock, 100) / 100` comes from a single read-only float32 table used by both
`encode_board` and the torch unpacker, so the CPU and GPU paths agree bit for bit.

## 2026-11-24: Packed record details

**Decision.**
- The dtype is unaligned and little-endian, so a record is 106 bytes rather than the
  ~108 first estimated.
- `ply` comes from the original board, because `mirror()` shifts `Board.ply()` for
  Black-to-move positions.
- `pack_into` rejects moves that are not legal in the board being packed. This makes the
  legacy mislabelling (the position after the move, labelled with that move) impossible,
  at about 17 µs per record including the check.
- In torch, bitboards are int64 tensors holding the uint64 bits unchanged, because torch's
  uint64 support is limited. Arithmetic right shifts still give the right bit after `& 1`.
- Torch is imported lazily inside the torch-facing functions, so preprocessing workers that
  only pack records never load it.
- `batch_from_records` copies into freshly allocated arrays. `np.ascontiguousarray` can
  return the strided field view itself (for a single record), which torch rejects.

## 2026-11-24: Value targets in `encoding/value.py`

**Decision.** The result-to-WDL mapping (`wdl_from_result`) and the class constants live in
their own small module next to the other encodings, not in the data pipeline. Packing,
training and search share one definition of "win for the side to move". Unfinished games
(`*`) raise instead of being guessed.

## 2026-11-24: Property-test settings

**Decision.** Hypothesis runs 200 examples per property with no deadline. In CI (the `CI`
environment variable is set) the profile is derandomized, so a failure reproduces exactly;
locally it explores new examples. Generated positions come from random playouts, with
every move drawn through hypothesis, from the standard start and from promotion-,
castling- and en-passant-rich positions.
