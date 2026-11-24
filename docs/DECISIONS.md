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
