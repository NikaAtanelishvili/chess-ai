# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `chessnet` package skeleton with a `chessnet` command-line entry point (`--help`, `--version`).
- uv-managed project (`pyproject.toml`, `uv.lock`, Python 3.11 pin) with CPU PyTorch by default.
- CI: ruff, mypy and tests on Python 3.11 and 3.12 with a coverage gate; nightly slow tests;
  gitleaks secret scanning; pull request title check.
- pre-commit hooks, including a Conventional Commits message check.
- Audit of the legacy engine (`docs/AUDIT.md`) and a decision log (`docs/DECISIONS.md`).

### Changed

- Legacy code moved unchanged from `src/` to `legacy/`; it is kept only as reference until
  the rebuild replaces it.

### Removed

- The legacy PyQt GUI.
- Tracked Lichess PGN dumps (still available in history and in the `legacy-v0` tag).
