# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `chessnet train`: fp16 mixed-precision training on CUDA with warmup and cosine decay,
  validation on a fixed subset, wall-clock and end-of-epoch checkpoints, exact resume
  (including the position within an epoch), and a clean stop on SIGTERM or
  `--max-wall-time`. Logs go to the console, CSV and TensorBoard, with W&B opt-in.
  Validation metrics include the White/Black accuracy split. `compile` and `channels_last`
  config flags select `torch.compile` and the NHWC memory format.
- `scripts/benchmark_train.py`: training throughput per model, batch size, `torch.compile`
  and channels_last.
- Kaggle training kernel (`kaggle/train/`): GPU tests first, then a smoke session or the
  main run, continuing from a previous session's output. On a two-GPU machine it also
  trains an independent second run (the new `wide` preset) on the other GPU, sized by a
  short benchmark and isolated so that it cannot stop or slow the main run.
- `wide` model preset (10 blocks x 192 filters, 7,572,332 parameters).
- `chessnet.train.budget`: run-time estimates that pick the largest number of epochs that
  fits a wall-time budget.
- `scripts/benchmark_train.py --json` writes the measurements for programs, and every row
  now includes the validation (forward-only) speed.

- Search: `NetworkEvaluator` (priors over legal moves only, value for the side to move, an
  LRU cache keyed by the network input) and PUCT MCTS with batched leaf evaluation, virtual
  loss, first-play urgency, terminal scoring (repetitions over the real game history) and
  tree reuse.
- Players behind one `Player.choose(board)` interface: `RandomPlayer`, `PolicyPlayer` and
  `MCTSPlayer` (playouts or time per move, optional temperature for the first plies).
- `scripts/benchmark_search.py` and the Kaggle kernel `kaggle/search/`: search speed on CPU
  and GPU with the trained base network.
- UCI engine (`chessnet uci`, and `chessnet-uci` for GUIs): `uci`, `isready`, `ucinewgame`,
  `position`, `go` (nodes, movetime, clocks, infinite) with `stop` from a worker thread,
  options (Weights, Simulations, CPuct, Mode, LeafBatch, Device, MoveOverhead), `info` lines
  with centipawn scores, and time management that does not flag at short time controls.
- `chessnet play`: a game against the engine in the terminal.
- `chessnet export-onnx`: export a checkpoint's network to ONNX.
- MCTS: an external stop event, a node limit, progress callbacks, and the chosen move's value.
- `chessnet eval --suite quick|full`: matches over 50 balanced openings with alternating
  colours (W/D/L, score, Elo with a 95% interval, PGNs), against a random mover, the policy,
  Stockfish at its weakest settings and a second network; Lichess puzzle accuracy by rating
  bucket; `--estimate` for wall time; results in `eval/results/` and `docs/MODEL_CARD.md`.
- A 500-puzzle Lichess sample in `tests/fixtures/puzzles/`, and the Kaggle kernel
  `kaggle/eval/` for the full suite.
- `PolicySafePlayer`: the policy plus three rules (take a mate in one; avoid moves that
  stalemate or allow a threefold-repetition claim while others exist). UCI `Mode policy` and
  `chessnet play --mode policy` now use it; the evaluation reports both it and the raw
  policy against a random mover. `chessnet eval --only` adds matches to earlier results.

### Changed

- The Kaggle preprocessing kernel installs v0.1.0. Kaggle CLI setup is documented with API
  tokens (`~/.kaggle/access_token` or `KAGGLE_API_TOKEN`).
- Development Python is 3.13, matching Kaggle's GPU image; CI tests 3.11 to 3.13.
- Training defaults follow the T4 benchmark: batch 2048 with `torch.compile` on and
  channels_last off. The main Kaggle config trains the base model for 5 epochs.

### Fixed

- `go depth N` was ignored; it now limits the search to `10 x N^2` playouts, so a GUI's
  depth setting changes the engine's strength.
- `uv run kaggle` crashed because the Kaggle CLI imports `six` without declaring it; `six`
  is now in the `dev` group, and CI checks that the CLI starts.

## [0.1.0] - 2026-11-25

First release of the rebuild: board and move encoding, the Lichess data pipeline and the
policy/value network. There are no trained weights yet.

### Added

- Policy/value network: SE-ResNet trunk with a convolutional policy head (in move-index
  order) and a WDL value head; `tiny`, `small` and `base` presets.
- Checkpoints with full training state that refuse weights from another encoding version,
  an inference wrapper, and ONNX export (`onnx` extra).
- `chessnet preprocess`: streams a Lichess `.pgn.zst` dump from a URL or a file (resuming
  dropped connections), filters games on their raw headers, parses survivors in parallel,
  splits by game, and writes globally shuffled shards with a `manifest.json`. A Kaggle CPU
  kernel runs it.
- `ShardDataset`: memory-mapped shards yielding packed batches for training.
- Encoding layer (`ENCODING_VERSION` 1): side-to-move board planes, the 4672-way
  AlphaZero move index with mirroring and legal-move mask, win/draw/loss targets, and a
  106-byte packed record format that unpacks to the same planes on CPU or GPU.
- `chessnet` package skeleton with a `chessnet` command-line entry point (`--help`, `--version`).
- uv-managed project (`pyproject.toml`, `uv.lock`, Python 3.11 pin) with CPU PyTorch by default.
- CI: ruff, mypy and tests on Python 3.11 and 3.12 with a coverage gate; nightly slow tests;
  gitleaks secret scanning; pull request title check.
- pre-commit hooks, including a Conventional Commits message check.
- Audit of the legacy engine (`docs/AUDIT.md`) and a decision log (`docs/DECISIONS.md`).
- MIT license.

### Changed

- Legacy code moved unchanged from `src/` to `legacy/`; it is kept only as reference until
  the rebuild replaces it.

### Removed

- The legacy PyQt GUI.
- Tracked Lichess PGN dumps (still available in history and in the `legacy-v0` tag).

[Unreleased]: https://github.com/NikaAtanelishvili/chess-ai/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/NikaAtanelishvili/chess-ai/releases/tag/v0.1.0
