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

## 2026-11-25: Nightly property runs

**Context.** The derandomized pull-request profile always tries the same 200 examples.
**Decision.** A `nightly` hypothesis profile (2000 examples, fresh random seeds, a
reproduction blob printed on failure) runs every property test in the nightly workflow:
`pytest -m hypothesis --hypothesis-profile=nightly`. Hypothesis marks every `@given` test
with `hypothesis` automatically. Pull requests keep the fixed-seed profile.

## 2026-11-25: GPU tests and the Kaggle parity check

**Decision.** Tests that need CUDA carry the `gpu` marker. Without a GPU they are skipped,
unless `--require-gpu` is passed, in which case they run and fail. `pytest -m gpu
--require-gpu` therefore passes only on a machine where every GPU test passed.
**Follow-up.** The first Kaggle training kernel checks out the pinned commit and runs
`pytest -m gpu --require-gpu` before training starts (the tests are not part of the
installed package). This exercises the GPU unpacking path, which CI cannot.

## 2026-11-25: Sign-safe bitboard handling

**Context.** In torch, bitboards are int64, so a piece on h8 (bit 63) makes the value
negative.
**Decision.** Bits are always extracted with shifts and `& 1`, never with sign-dependent
tests such as `> 0`. Corner-square tests (a1, h1, a8, h8; mover and opponent; both
colours) and extreme bitboards (only bit 63 set, all bits set) guard this on the CPU path.

## 2026-11-25: Data pipeline structure

**Decision.** `chessnet.data` holds `pgn_stream` (sources and game splitting), `filters`,
`shards` (splits, shuffle, manifest) and `dataset` (loading). It also holds `pipeline`, the
orchestrator behind `chessnet preprocess`, which the original layout did not list. Two
top-level helpers were added: `chessnet.config` (pydantic models loaded from YAML) and
`chessnet.provenance` (the git commit for manifests and, later, checkpoints).

## 2026-11-25: Streaming sources

**Decision.**
- HTTP uses the standard library (`urllib`) rather than `requests`, so there are no extra
  dependencies.
- A dropped connection, or a body shorter than its `Content-Length`, is resumed with a
  `Range: bytes=<offset>-` request after an exponential backoff (1 s, 2 s, 4 s, ... up to
  8 retries). A server that ignores `Range` stops the run instead of silently restarting.
  Client errors (4xx except 408/429) are not retried. (`http.client` reports a premature
  end of the body as a normal end of file, so the length is checked explicitly.)
- Compression is detected from the zstd magic number, so `.pgn.zst` and plain `.pgn` both
  work. Lichess dumps start with a zstd *skippable* frame and consist of many frames, so
  skippable-frame magics count as zstd too, and the decompressor reads across frames.
  (python-zstandard's default stops at the end of the first frame.) Windows up to 2 GiB
  are allowed.
- Input that shows no game within its first 64 MiB is refused, and the skipped bytes are
  not buffered.
- Games are split on raw bytes at `\n[Event `, and the header filter reads raw bytes too.
  Most games are rejected without decoding text or running python-chess.

## 2026-11-25: Parallel parsing and determinism

**Decision.** Games that pass the header filter go in chunks of 64 to a `spawn`-started
process pool, with at most two chunks per worker in flight (`Pool.imap` would read the
whole dump into memory). Results are consumed in submission order, and bucket assignment
uses seeded generators, so the same source, config and seed give byte-identical shards.
`max_games` stops at exactly that many kept games. The byte and scan counters in the
manifest include the few chunks read ahead. `--workers 0` parses in-process, for tests and
debugging.

## 2026-11-25: Filter semantics

**Decision.**
- Results other than `1-0`, `0-1` and `1/2-1/2`, and games with `Variant`, `FEN` or
  `SetUp` headers, are always rejected; neither is configurable.
- Unknown ratings (`?`) are rejected even with a rating floor of 0.
- Each rejected game is counted under its first failing check, in this order: variant,
  result, termination, time control, rating, URL. Parse errors and games shorter than
  `min_plies` are counted by the workers.

## 2026-11-25: Split hash

**Decision.** The split is decided by BLAKE2b (8 bytes) of the full `Site` string, mod 100:
0-97 train, 98 val, 99 test. Python's built-in `hash()` is randomised per process, so it
cannot be used. Recent dumps use `https://lichess.org/game/<id>` URLs where older ones used
`https://lichess.org/<id>`, so the same game would land in different splits across URL
formats. That only matters when mixing months with different formats, which the pipeline
does not do.

## 2026-11-25: Shuffle and shard layout

**Decision.**
- Pass 1 appends records to random bucket files per split: `num_buckets` for train, and
  proportionally fewer for val and test, at least one each.
- Pass 2 permutes each bucket, deletes it at once, and cuts the stream of permuted buckets
  into shards of exactly `shard_size` positions (only the last shard of a split is
  shorter). Uniform shard sizes keep loading simple.
- Peak disk use is about one dataset plus one bucket, and peak memory about one bucket plus
  one shard.

## 2026-11-25: Manifest and failure handling

**Decision.**
- The manifest is a pydantic model, written atomically and **last**, so a directory without
  one is an incomplete run.
- It records the encoding version and record layout, config, source and byte counts,
  per-reason rejection counts, per-split game and position counts, a SHA-256 per shard,
  the package version, the git commit (from the installer record for `pip install
  git+...`, otherwise from git, with `-dirty` for uncommitted changes) and timings.
- Reading data built with another encoding version raises.
- If any move was unrepresentable, the run fails before writing a manifest.

## 2026-11-25: Training data loader

**Decision.**
- `ShardDataset` is an `IterableDataset` over memory-mapped shards, yielding `PackedBatch`
  objects (CPU tensors) of consecutive records.
- Each epoch permutes the shard order and starts every shard at a random offset below the
  batch size, both derived from `(seed, epoch)`.
- With DataLoader workers, each worker reads a disjoint subset of the shards.
- Pinned memory and resuming mid-epoch are left to the training loop, which is where they
  are needed.

## 2026-11-25: Pipeline test fixture

**Decision.**
- The fixture is 20 real games from the October 2026 dump (CC0), anonymized: player
  names and game URLs are replaced and title headers dropped.
- Game URLs are chosen so that one kept game lands in each of val and test.
- A few headers are edited so that every rejected game fails exactly one check, and so that
  cases absent from the rated dump (casual, unfinished, Chess960) are covered.
- The expected outcome of every game is tabulated in `tests/fixtures/README.md`.

## 2026-11-25: Mirroring without copying the move stack

**Context.** python-chess's `Board.mirror()` copies the board including its whole move stack,
then clears the stack. On a board that carries its game history (replayed games in the data
pipeline, game positions in search), each mirror cost O(game length).
**Decision.** `perspective_board` copies with `stack=False` and mirrors in place
(`apply_mirror`). The result is identical, and the preprocessing pipeline became 1.75x faster
(533 instead of 305 kept games/s on the development machine). A regression test checks
that the stack is not copied.

## 2026-11-25: Rated games are assumed from the source (owner decision)

**Context.** The first filter kept games whose Event contained `Rated`. Lichess names
rated Swiss games `Blitz swiss ...`, so that check dropped rated games: 0.4% of the dump,
about 0.2-0.3% of the usable data.
**Decision.** Rated-ness is not checked. The source must be a rated-games dump
(`lichess_db_standard_rated_*`); every manifest records this assumption under
`assumptions`, and the standard-variant check stays. A real rated Swiss game is part of the
pipeline fixture. Casual games in some other source would be kept, which is why the
assumption is written down.

## 2026-11-25: Commit provenance for local installs

**Context.** The Kaggle kernels install the package from a clone (`pip install
/tmp/chessnet`). The installer then records the source directory, not a VCS commit, so
manifests would have had no git commit.
**Decision.** `git_commit()` asks git in the directory a local install came from before
falling back to the package's own source tree.

## 2026-11-25: Network details

**Decision.**
- Every preset uses Leela-style squeeze-excitation (a learned per-channel scale and bias,
  reduction 4), not only `base`. The presets then differ in size only, and the tiny model
  in the tests exercises the same code paths as the Kaggle model.
- Both heads have BN and ReLU after their first convolution.
- The policy head is fully convolutional: the `(73, 8, 8)` output is flattened directly
  into the move-index order. A dense 4672-way layer (as in the legacy code, 21.8M
  parameters) is avoided.
- Parameter counts per preset are pinned by a test, so architecture changes are deliberate.

## 2026-11-25: Checkpoint format

**Decision.**
- Checkpoints are plain dictionaries with a format number, loaded with `weights_only=True`.
  RNG states are therefore stored as plain values (the NumPy state as a list), not as
  pickled objects.
- Files are written to a temporary name and renamed, so a session killed mid-save (Kaggle's
  12-hour limit) never leaves a corrupt checkpoint.
- The model config is always embedded, so `load_network` needs nothing but the file.
- `EncodingVersionError` now lives in `chessnet.encoding` and is raised both for datasets and
  for checkpoints of another encoding version.

## 2026-11-25: Inference and ONNX

**Decision.**
- Inference code (`InferenceModel`, `export_onnx`) lives in `model/inference.py`, a module
  the original layout did not list.
- ONNX export uses torch's current dynamo-based exporter with a dynamic batch dimension,
  opset 18 and `external_data=False` (one self-contained file).
- Its dependencies (`onnx`, `onnxscript`, `onnxruntime`) form an optional `onnx` extra.
- The ONNX parity and `torch.compile` tests are marked slow and run nightly.

## 2026-11-25: Kaggle CLI dependencies and authentication

**Context.** kaggle 2.2.5, the latest release, imports `six` without declaring it, so
`uv run kaggle` crashed once nothing else pulled `six` in. Kaggle now issues API tokens
instead of `kaggle.json` keys.
**Decision.** `six` is in the `dev` group, with a comment to drop it once kaggle declares it,
and CI runs `uv run kaggle --version`, which needs no credentials. The docs describe the
token in `~/.kaggle/access_token` or `KAGGLE_API_TOKEN` as the way to authenticate, which
2.2.5 checks first. Token files are gitignored.
