# chessnet

An AlphaZero-style chess engine: a residual CNN predicts move probabilities and the
expected outcome, and Monte Carlo tree search (PUCT) uses both to choose moves. Unlike
AlphaZero it is trained with supervised learning on rated human games from the
[Lichess open database](https://database.lichess.org/), not by self-play.

> **Status:** `0.2.0.dev0`. The network is trained (55% validation top-1 on 2000+ rated
> games), and the engine plays through MCTS in the terminal or in any UCI chess GUI. The
> previous implementation is preserved under the `legacy-v0` tag; see
> [docs/AUDIT.md](docs/AUDIT.md) for why it was replaced.

## Development setup

Requires [uv](https://docs.astral.sh/uv/). uv installs the pinned Python (3.13) if needed.

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

Data preparation and training run on Kaggle through the Kaggle CLI (`uv run kaggle ...`).
It authenticates with an API token from <https://www.kaggle.com/settings/api>, saved to
`~/.kaggle/access_token` or exported as `KAGGLE_API_TOKEN`; see
[docs/TRAINING.md](docs/TRAINING.md#kaggle-cli-setup).

## Play against chessnet

The engine needs a trained network. The one in use, `checkpoints/session2/base-best.pt`,
is not in git (see [docs/TRAINING.md](docs/TRAINING.md#checkpoints) for its source and
SHA-256). Put it there, or point the engine elsewhere with `--weights`, the `Weights` UCI
option or the `CHESSNET_WEIGHTS` environment variable.

### In the terminal

```bash
uv run chessnet play                     # you play White
uv run chessnet play --color black       # or random
uv run chessnet play --movetime 3        # 3 seconds per engine move (default: 800 playouts)
uv run chessnet play --mode policy       # instant moves (policy-safe), no search
```

Type moves in SAN (`Nf3`, `O-O`) or UCI (`g1f3`); `moves` lists the legal moves, `undo`
takes back your last move, `resign` ends the game.

### In a chess GUI (UCI)

`uv sync` creates the engine program `.venv/bin/chessnet-uci` in the repository. Point the
GUI at its full path, which this prints (run it in the repository):

```bash
realpath .venv/bin/chessnet-uci      # e.g. /home/you/chess-ai/.venv/bin/chessnet-uci
```

It needs no arguments and no activated environment, and finds the network by itself. On a
laptop CPU it searches about 500 nodes per second, so 800 playouts take about 1.5 s per move.

**En Croissant (Ubuntu):**

1. Install En Croissant from its [releases page](https://github.com/franciscoBSalgueiro/en-croissant/releases)
   (`.deb` or AppImage).
2. Open **Engines** (bottom of the left sidebar), click **Add New** (in some versions a
   **+** button), choose the **Local** tab, click **Binary file**, and select
   the `chessnet-uci` path printed above. Save the engine, which is named
   `chessnet 0.2.0.dev0`.
3. To play: open a new tab with **+**, choose **Play Chess**, set the opponent to the
   engine **chessnet**, and limit it by **nodes** (for example 800), **time** or **depth**.
   Depth N means `10 x N^2` playouts: depth 9 is about the default 800, depth 20 is 4,000.

**Cute Chess (Ubuntu):**

1. Install Cute Chess (packages are on its [releases page](https://github.com/cutechess/cutechess/releases)).
2. **Tools > Settings > Engines**, then **+**. Set *Name* `chessnet`, *Command* the
   `chessnet-uci` path printed above, *Working directory* the repository, *Protocol* `UCI`,
   and click OK.
3. **Game > New**, set one side to *CPU* with *chessnet*, choose a time control, and start.

**UCI options:**

| Option | Default | Meaning |
|--------|---------|---------|
| `Weights` | `checkpoints/session2/base-best.pt` | network checkpoint |
| `Simulations` | 800 | playouts per move when the GUI sets no node or time limit |
| `CPuct` | 2.0 | weight of the prior in PUCT exploration |
| `Mode` | `mcts` | `mcts` (search) or `policy` (policy-safe: instant, see below) |
| `LeafBatch` | 32 | leaves per network call (fastest on the CPUs measured) |
| `Device` | `auto` | `auto`, `cpu` or `cuda` |
| `MoveOverhead` | 100 | milliseconds kept back per move for GUI and system lag |

`Mode policy` plays **policy-safe**: the network's most likely move, except that it takes
a mate in one when it has one, and, when the network rates its position better than level,
avoids moves that stalemate or let either side claim a threefold repetition while it has
other moves. Level or worse, a draw is welcome. The evaluation's `policy` player stays the
raw network, for an honest baseline.

The engine understands `go nodes`, `movetime`, `depth`, `wtime`/`btime`/`winc`/`binc`/
`movestogo`, and `infinite` with `stop`. MCTS has no fixed depth, so `go depth N` runs
`10 x N^2` playouts (at most a million): 10 at depth 1, 810 at depth 9, 4,000 at depth 20.
With a clock it thinks for `remaining / 30 + 0.8 x increment`,
less the move overhead, and never more than half the remaining time. Scores are converted
from the network's expected outcome `v` with Lichess's win-chance model,
`cp = ln((1 + v) / (1 - v)) / 0.00368208` (`v = 0.18` is about one pawn).

## Evaluate

```bash
uv run chessnet eval --suite quick              # about 10 minutes on a laptop CPU
uv run chessnet eval --suite full --estimate    # times a few games, estimates the full suite
uv run chessnet eval --suite full --workers 2 --compare checkpoints/session2/wide-best.pt \
    --puzzles data/puzzles/lichess_db_puzzle.csv.zst
```

The quick suite plays the raw policy and policy-safe against a random mover and MCTS (400
playouts) against the policy, and scores 500 Lichess puzzles. The full suite plays more games, adds MCTS (800
playouts) against Stockfish at its lowest `UCI_Elo` and at `Skill Level` 0, optionally a
second network against the first, and 2,000 puzzles. Games start from 50 balanced
openings, each played with both colours. Results go to `eval/results/` (JSON and PGNs),
and a summary table to [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

- **Stockfish** is optional; its matches are skipped without it. On Ubuntu:
  `sudo apt install stockfish`.
- **Puzzles:** 500 are committed in `tests/fixtures/puzzles/`. The full Lichess puzzle
  database (CC0, 310 MB) is not:
  `curl -L -o data/puzzles/lichess_db_puzzle.csv.zst https://database.lichess.org/lichess_db_puzzle.csv.zst`
- The full suite takes hours on a laptop. The Kaggle kernel `kaggle/eval/` runs it, or
  chosen parts of it, on a CPU machine (or a T4 machine): set `CHESSNET_REF` and `ONLY` in
  `kaggle/eval/eval.py`, then `uv run kaggle kernels push -p kaggle/eval`.
- Long runs cannot hang: a stuck game or engine is killed and counted as an error, a
  heartbeat line appears every minute, each match is saved as soon as it ends, and
  `--max-wall-time 6h` stops cleanly with what has finished. `--only <part>,...` runs
  single matches or puzzle runs (such as `puzzles-mcts-800`) and merges them into earlier
  results.

## Contributing

Commits follow [Conventional Commits](https://www.conventionalcommits.org/) using the types
`feat`, `fix`, `refactor`, `perf`, `style`, `test`, `docs`, `build`, `ops` and `chore`
(enforced by a commit-msg hook). Design decisions are logged in
[docs/DECISIONS.md](docs/DECISIONS.md).

## License

[MIT](LICENSE).
