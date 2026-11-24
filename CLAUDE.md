# chessnet: notes for Claude Code

## Commands

- Env: uv-managed `.venv/` (standard venv; activating it is fine). Deps only via
  `uv add` / `uv add --group dev`; `uv.lock` is committed. Never pip-install ad hoc.
- Install: `uv sync --all-groups --all-extras`
- Lint/format: `uv run ruff check . && uv run ruff format .`
- Types: `uv run mypy src`
- Fast tests: `uv run pytest -m "not slow"`
- All tests: `uv run pytest`
- Kaggle CLI: `uv run kaggle ...`; it authenticates with a token in `~/.kaggle/access_token`
  or `KAGGLE_API_TOKEN`. Never put tokens in the repository.

## Invariants (never break these)

- All network inputs/outputs are from the SIDE-TO-MOVE perspective. Black-to-move
  boards are encoded via `board.mirror()`; moves via `chess.square_mirror`.
- Encoder and decoder of the 4672 policy index are one module (`encoding/moves.py`).
  Never re-implement the index anywhere else.
- MCTS backup negates the value every ply.
- Any change to planes/move index/packed format => bump `ENCODING_VERSION`,
  regenerate the dataset, retrain. Old checkpoints must fail to load.
- Train/val/test split is by GAME (hash of the Site URL), never by position.

## Workflow

- Commits follow Conventional Commits (qoomon cheatsheet):
  `<type>(<scope>): <imperative lowercase description>`. Types: feat, fix,
  refactor, perf, style, test, docs, build, ops, chore. CI pipeline changes are `ops(ci)`.
  Any ENCODING_VERSION bump is a breaking change (`!` + `BREAKING CHANGE:` footer).
- One branch per feature; CI must be green before merge.
- Bug fix = failing regression test first.
- Record non-obvious decisions in docs/DECISIONS.md.
- Never commit data, checkpoints, or secrets.
- Tracked files may be deleted/rewritten freely (history keeps them). Ask before
  deleting untracked or gitignored files (datasets, checkpoints, logs).
- Don't start multi-hour data downloads or training runs; the owner launches those.
