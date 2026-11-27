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

## 2026-11-25: Python 3.13 to match Kaggle's GPU image

**Context.** Kaggle's GPU image is now built on Colab's 2026-10-27 runtime, whose Python is
3.13 (`/usr/local/lib/python3.13/dist-packages` in `Kaggle/docker-python`).
**Decision.** `.python-version` is now 3.13, and CI tests 3.11, 3.12 and 3.13.
`requires-python` stays `>=3.11`, and mypy still checks against 3.11 syntax. The full suite
passes on 3.13. This settles the earlier follow-up on the Python pin.

## 2026-11-25: Kaggle GPU choice

**Context.** Kaggle has retired the P100; the CLI warns that P100 requests run on a T4.
**Decision.** There is only a T4 training config. The training kernel requests
`machine_shape: NvidiaTeslaT4` (two T4s) and trains on one GPU. Multi-GPU training stays out
of scope unless the benchmark shows the data loader idle and one GPU saturated.

## 2026-11-25: Training loop details

**Decision.**
- **Precision:** fp16 autocast with a GradScaler on CUDA only, fp32 on the CPU (T4s handle
  bf16 poorly).
- **Optimisation:** biases and normalisation weights get no weight decay. Warmup plus cosine
  was chosen over one-cycle: one schedule is enough, and it is easy to resume.
- **Schedule length:** fixed when a run starts (`total_steps`, stored in every checkpoint),
  so a multi-session run keeps one schedule. Changing `epochs` later means a new run.
- **Validation subset:** the first `val_batches` batches of the unshuffled val split, the
  same positions every time.
- **Checkpoint names:** `checkpoints/step-NNNNNNNNN.pt` (last 3 kept) and `best.pt`.
  `--resume auto` picks the newest rolling checkpoint in the run directory and any
  `--resume-from` directories, never `best.pt`.
- **Resume checks:** resuming refuses a checkpoint from a different dataset (by manifest
  hash) or a different model.
- **Data position:** `ShardDataset.start_after` skips the batches already trained on, the
  way a DataLoader with N workers delivers them (round-robin over workers, passing over
  workers that have run dry). Worker roles are rotated so a fresh DataLoader continues with
  the worker that was due next. Skipped data is never read.
- **DataLoader workers** start through `forkserver`: forking a process that runs threads
  (TensorBoard, torch) can deadlock children.
- **No GPU syncs per step:** training metrics are summed on the GPU and read back only at
  log steps.
- **No pinned memory:** packed batches are about 150 KB, so pinning would not help.
- **Logging:** W&B is opt-in, imported lazily, and never fatal. TensorBoard needs the
  `train` extra, which the Kaggle kernel installs.

## 2026-11-25: Kill-and-resume test placement

**Decision.** The kill-and-resume equivalence test (60 steps without interruption vs. 30 +
resumed 30; per-step losses and final weights must match) runs on every CI run without
DataLoader workers. The variant with 2 workers takes another 12 s and runs nightly. Resuming
with workers is also unit-tested against a real DataLoader with 2 and 3 workers in the fast
suite.

## 2026-11-25: Kaggle training kernel

**Decision.**
- The kernel installs a pinned commit with the `train` extra plus pytest and hypothesis, and
  runs `pytest -m gpu --require-gpu` before anything else.
- It finds the dataset by globbing `/kaggle/input/**/manifest.json`, and previous runs by
  `run-base/runs.jsonl` and `run-wide/runs.jsonl`, failing on zero (dataset) or several
  matches.
- `MODE = "smoke"` benchmarks and runs the tiny model. `MODE = "train"` copies each run
  directory of an attached previous session into `/kaggle/working` and resumes it, so each
  session's output holds both runs so far. (Session 2 added the second run; see below.)

## 2026-11-25: Training speed options

**Context.** `torch.compile` and the channels_last memory format may or may not speed up
fp16 training on a T4, depending on the model size and the PyTorch build. With sessions
capped at 12 h, the choice should be measured, not guessed.
**Decision.**
- Both are `TrainConfig` flags (`compile`, `channels_last`), applied by one helper
  (`prepare_network`) that the training loop and the benchmark share. Their defaults are
  the fastest combination measured on Kaggle's T4 in session 1: compile on, channels_last
  off (channels_last helps an eager model but slows a compiled one). CPU tests turn
  compiling off; one slow test compiles.
- They never reach a checkpoint: the weights stay in the uncompiled network, and loading
  copies values whatever the memory format, so a run can change options between sessions.
- Validation always runs the eager network. It is a small share of the time, and calling
  the compiled module in eval mode would compile a second graph.
- The benchmark records a compiler failure in its table and carries on, so a broken
  compiler cannot end the smoke session.

## 2026-11-25: Main run settings

**Context.** Session 1 measured the base model on a T4 at 10,639 positions/s with batch 2048
and `torch.compile`, against 10,058 at batch 1024.
**Decision.**
- Batch 2048 (also the `TrainConfig` default): 6% faster than 1024, and it fits 5 epochs
  into one session instead of 4.
- The learning rate stays at 1e-3 without batch-size scaling: AdamW at 1e-3 is a safe
  setting at this batch size, and there is no budget for tuning runs. Warmup stays at 2000
  steps (4.1M positions).
- Validation every 2500 steps on 25 batches: the same 51,200 positions, every 5.1M
  positions, as at batch 1024.
- Epochs: the largest whole number whose estimated run time leaves at least 1 h of the
  11h15m wall-time budget. That is 5 epochs: 10.18 h, counting validation, checkpoints,
  start-up and compilation from measured speeds (`chessnet.train.budget`).
- One DataLoader worker instead of two, since the second run shares the machine's 4 vCPUs:
  one worker delivers about 180,000 positions/s, 17 times what base consumes.

## 2026-11-25: A second, isolated run on the second T4

**Context.** Every T4 session has two GPUs, and the main run uses one. The owner chose no DDP
for this run (see the backlog entry below) and asked for an independent run on the idle GPU.
**Decision.**
- The second run trains `wide` (10 blocks x 192 filters, 7.6M parameters) with the main
  run's settings, for an equal-time comparison with base at 5 epochs.
- Its epochs are chosen during the session from a 2-minute benchmark on GPU 1, with the same
  estimate as the main run's (`chessnet.train.budget`) but a 1.5 h margin, because its speed
  is measured only minutes before it starts. Its wall time ends with the main run's.
- Isolation from the main run:
  - separate processes, GPUs, output directories and logs;
  - the second run starts only after the main run has logged training progress;
  - it runs at the lowest CPU priority, with one compiler thread, one OpenMP thread and its
    own compiler caches;
  - one DataLoader worker per run;
  - the kernel never signals the main run, and stops a second run only once it is past its
    deadline (SIGTERM, then SIGKILL).
- `run-wide/epochs.json` records the measured speed and the chosen epochs. A resumed second
  run keeps its `run-wide/config.yaml`, so its schedule does not change.

## 2026-11-25: Backlog: DDP for the next, bigger-model run

**Context.** For session 2 the owner chose two independent runs over DDP. The next run, with
a bigger model, should use both T4s for one job.
**Decision (to implement then).**
- **Expected gain:** about 1.8x with 1024 positions per GPU, which keeps the hyperparameters
  of a 2048 batch (2 x 10,058 positions/s for base, at about 95% efficiency). Up to 1.95x
  with 2048 per GPU, but that 4096 batch needs an untested learning rate. Each step
  all-reduces the gradients (14 MB for base, 30 MB for wide) over PCIe in 2-3 ms, against
  a 100-190 ms step. In one 11h15m session with a 1 h margin that is 8 epochs of base instead
  of 5, or 5 epochs of a model with about 1.7 times base's compute (for example 18 blocks x
  128 filters, about 6M parameters).
- **Shape:** the kernel launches `torchrun --standalone --nproc_per_node=2`, with an NCCL
  process group and one GPU per rank. `ShardDataset` gives each rank a disjoint,
  deterministic share of every epoch's batches. Rank 0 alone logs, validates and writes
  checkpoints, and the logged metrics are all-reduced.
- **Hang safety.** A rank waiting in a collective while the other has stopped would idle
  the session until Kaggle's 12-hour limit and lose the final checkpoint. Against that:
  - every stop decision is collective: each step all-reduces a stop flag (SIGTERM, SIGINT,
    wall time, step limit), so all ranks stop after the same step;
  - every rank runs the same number of steps per epoch, computed from the shard sizes before
    the epoch starts; surplus batches are dropped, never waited for;
  - rank 0's validation and checkpoint writes sit between barriers, so the other rank waits
    at a barrier, never inside a gradient all-reduce;
  - the NCCL process group has a timeout (about 10 minutes) and asynchronous error handling
    (`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`), so a hang becomes a crash; rolling checkpoints
    bound the loss to 20 minutes of training;
  - an exception on any rank ends the whole job (torchrun's default), and the kernel reports
    it.
- **Exact resume:** checkpoints record the global step, and each rank derives its share of
  the remaining batches from it. A CPU test with two gloo processes checks kill-and-resume
  equivalence in CI.
- **Verification before a long run:** one short Kaggle session of about 0.5 h (GPU tests,
  two-GPU throughput with `torch.compile`, a short run stopped by `--max-wall-time` and
  resumed). Budget 1 h in case it finds a bug.

## 2026-11-26: Network for search and play

**Context.** Session 2 trained base (5 epochs) and wide (2 epochs) for about the same GPU
time. Both reached about 55% validation top-1, with wide ahead by 0.24 points, which is
within the noise of the validation subset.
**Decision.** Search, the UCI engine and evaluation use base, `checkpoints/session2/base-best.pt`
(SHA-256 `370248af...` in `docs/TRAINING.md`). It is as accurate as wide and about twice as
fast per evaluation, which buys more search in the same time, especially on a CPU. Wide
stays available for a fixed-node comparison in evaluation.

## 2026-11-26: Search conventions

**Decision.**
- **Value perspective:** a node keeps its value for the player who moved into it, so
  selection reads children's mean values directly and backup negates at every ply.
- **Terminal rules:** threefold repetition (`board.is_repetition(3)`) and the 50-move rule
  count as draws inside the tree, the claimable versions, because a player facing a loss
  would claim them. The root is never terminal while it has legal moves: an engine must
  still move when a draw could be claimed.
- **History:** the search works on a copy of the game board with its full move stack, so
  repetitions include the game before the root. A tree is reused only if the new board
  continues the same game, because terminal flags in the tree depend on that history.
- **Budget:** `simulations` counts the playouts added by one search, not the root's total, so
  tree reuse never shortens a search.
- **Batches:** a leaf selected twice in one batch ends the batch; no playout waits on a
  duplicate, and every batch makes progress.
- **Cache key:** the network input, including the halfmove clock up to 100. A Zobrist hash
  alone would mix positions whose clock feature differs, and the key is exact.
- No Dirichlet noise: it serves self-play exploration, which this engine does not do.

## 2026-11-26: UCI engine details

**Decision.**
- **Entry point for GUIs:** a `chessnet-uci` console script. GUIs such as En Croissant start
  an engine from a binary path without arguments, so the engine finds its network by itself:
  the `Weights` option, else `CHESSNET_WEIGHTS`, else `checkpoints/session2/base-best.pt`
  in the checkout (the package is installed in editable mode).
- **Lazy loading:** `uci` must be answered quickly, so torch and the network load on
  `isready` or at the first `go`.
- **Score mapping:** Lichess's win-chance model, `cp = ln((1 + v) / (1 - v)) / 0.00368208`.
  It is a standard logistic mapping, and it was fitted to the same population of games.
- **Depth:** reported as the principal variation's length. MCTS has no depth limit of its
  own, so `go depth N` becomes a playout budget, `10 x N^2` (capped at a million). It
  grows with N, puts depth 9 at about the default 800 playouts, and keeps the depths GUIs
  offer (up to about 20 to 30) at seconds per move on a CPU. (2026-11-27; it was ignored
  before.)
- **Defaults:** `LeafBatch` 32, the fastest batch on both CPUs measured; `Simulations` 800,
  about 1.5 s a move on a laptop CPU. `MoveOverhead` (100 ms) is an option, because the
  time lost per move depends on the GUI and the machine.
- **Memory bound:** search stops growing at half the memory available at start-up (about
  9 KB per playout). An infinite search keeps its result until `stop`. Storing children
  lazily, so that unvisited moves cost no node objects, would cut the tree's memory
  several times; that can come later.
- **Robustness:** errors (bad weights, illegal moves in `position`, malformed `go`) become
  `info string` lines, and the engine keeps running.

## 2026-11-27: Evaluation design

**Decision.**
- **Openings:** 50 balanced main lines (Stockfish 16, depth 20, within +-52 cp), each
  played with both colours. Without them, two deterministic players would replay one game.
- **Draws** end a game only when a repetition has happened or the 50-move rule applies.
  python-chess's `outcome(claim_draw=True)` also counts a repetition that one legal move
  could reach, which would end games a winning side is still pressing. This matches the
  search's own terminal rules.
- **Stockfish** gets 1 s per move, near the conditions its `UCI_Elo` is calibrated for,
  with one thread and 16 MB hash. Our MCTS plays a fixed number of playouts, as specified.
- **Elo intervals:** normal approximation of the score's per-game variance; Wilson's
  interval when all games had one result, which keeps a sweep's lower bound finite.
- **Puzzles:** first move only, with any mate accepted when the solution mates. Equal
  samples per rating bucket from well-established puzzles, so each bucket's accuracy has
  the same precision.
- **Where it runs:** the full suite takes hours on a laptop, so a Kaggle kernel runs it with
  4 parallel games. Results (JSON and PGNs) are committed under `eval/results/`, so the
  model card's numbers can be checked.

## 2026-11-27: Policy-safe player (owner decision)

**Context.** In the quick suite the raw policy scored 90% against a random mover, below the
98% target: 3 of 20 games were stalemates and one a threefold repetition, never a loss.
Without search the policy cannot tell that stalemating a lone king throws the win away.
**Decision.**
- The `policy` player stays the raw network, so the baseline measures the network honestly.
- `policy-safe` adds three rules: take a mate in one; otherwise skip moves that stalemate or
  leave a threefold repetition to claim (completed by the move, or reachable by the
  opponent's reply), unless every move does. It is no search: one ply of rules, applied in
  prior order.
- Both play the random mover in the suites. UCI `Mode policy` and `chessnet play --mode
  policy` use policy-safe, since a player wants the win.
- The Kaggle full suite was already running, so policy-safe's 100 games against the random
  mover were played on the laptop and merged into its results with `--only`.
- Refined the same day (owner decision): the draw rules apply only when the value head
  rates the side to move better (`v > 0`). Level or worse, a draw is welcome, so moves that
  stalemate or repeat are allowed. A mate in one is always taken. The results against the
  random mover were replayed with the refined player.

## 2026-11-27: Hang-proof evaluation runs

**Context.** The first full suite on Kaggle (2 x T4) finished three matches in 20 minutes
and then printed nothing for 8.5 hours, until it was stopped by hand. python-chess runs
each engine's event loop in a non-daemon thread. The worker processes (a
`ProcessPoolExecutor`) closed their players in an `atexit` handler, but Python joins
non-daemon threads before it runs `atexit` handlers, so a worker holding a Stockfish
process never exited, and the pool's shutdown after the first Stockfish match waited for
it forever. That match's results were still in memory; they were recovered from the log,
but its PGN was lost.
**Decision.**
- Workers close their players in a `finally` block, before the interpreter waits for
  threads. A regression test plays a match against an external engine in two workers.
- A small supervisor replaces the process pool. Each worker has its own pipes, so killing
  one cannot corrupt a shared queue, and leads its own process group, so killing it kills
  its engines. Limits: 5 minutes without a move, 1 hour per game or puzzle, the suite's
  `--max-wall-time`, and the engine call timeout of python-chess (move time + 30 s).
- A game that hits a limit is recorded with its reason and left out of the score, rather
  than retried: a retry could hang the same way, and the count of failures is reported.
- Retries of workers are bounded: after 4 in a row die without finishing anything, the
  rest of the run fails with the reason.
- A heartbeat line every 60 s, and results saved after every match.
- The evaluation kernel always runs with `--max-wall-time 11h` (the session limit is 12 h)
  and asks for a CPU machine unless the metadata is changed: the owner prefers the laptop
  or a Kaggle CPU kernel, and the GPU only when CPU time would be unreasonable.
**Alternatives.** Making python-chess's thread a daemon (not configurable); `os._exit` at
the end of each worker (skips closing the engines, which keep running); a timeout around
the pool's shutdown alone (a stuck engine in the middle of a game would still hang, and the
results would still be lost).
