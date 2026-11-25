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
  Validation metrics include the White/Black accuracy split.
- `scripts/benchmark_train.py`: training throughput per model and batch size.
- Kaggle training kernel (`kaggle/train/`): GPU tests first, then a smoke session or the
  main run, continuing from a previous session's output.

### Changed

- The Kaggle preprocessing kernel installs v0.1.0. Kaggle CLI setup is documented with API
  tokens (`~/.kaggle/access_token` or `KAGGLE_API_TOKEN`).
- Development Python is 3.13, matching Kaggle's GPU image; CI tests 3.11 to 3.13.

### Fixed

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
