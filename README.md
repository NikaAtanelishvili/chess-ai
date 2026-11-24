# chessnet

An AlphaZero-style chess engine: a residual CNN predicts move probabilities and the
expected outcome, and Monte Carlo tree search (PUCT) uses both to choose moves. Unlike
AlphaZero it is trained with supervised learning on rated human games from the
[Lichess open database](https://database.lichess.org/), not by self-play.

> **Status:** early rebuild (`0.1.0.dev0`). Nothing is playable yet. The previous
> implementation is preserved under the `legacy-v0` tag; see [docs/AUDIT.md](docs/AUDIT.md)
> for why it was replaced.

## Development setup

Requires [uv](https://docs.astral.sh/uv/). uv installs the pinned Python (3.11) if needed.

```bash
uv sync --all-groups --all-extras         # create .venv/ with all dependencies
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
uv run chessnet --help
```

To use the environment directly instead of `uv run`, activate it:

```bash
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows
```

Common commands:

| Task | Command |
|------|---------|
| Lint and format | `uv run ruff check . && uv run ruff format .` |
| Type check | `uv run mypy` |
| Fast tests | `uv run pytest -m "not slow"` |
| All tests | `uv run pytest` |

PyTorch is installed from the CPU wheel index by default. Dependencies are declared in
`pyproject.toml` and pinned in `uv.lock`; add them with `uv add` (or `uv add --group dev`).

## Contributing

Commits follow [Conventional Commits](https://www.conventionalcommits.org/) using the types
`feat`, `fix`, `refactor`, `perf`, `style`, `test`, `docs`, `build`, `ops` and `chore`
(enforced by a commit-msg hook). Design decisions are logged in
[docs/DECISIONS.md](docs/DECISIONS.md).
