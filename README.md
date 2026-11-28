# chessnet

An AlphaZero-style chess engine: a residual CNN predicts move probabilities and the
expected outcome, and Monte Carlo tree search (PUCT) uses both to choose moves. Unlike
AlphaZero it is trained with supervised learning on rated human games from the
[Lichess open database](https://database.lichess.org/), not by self-play. It plays in the
terminal or in any UCI chess GUI, and the trained network is released with the code.

> **Status:** `1.0.0`. The `base` network (10 residual blocks, 3.5M parameters) was trained
> on 77M positions from games between players rated 2000 or more; it predicts the move
> played 55% of the time on held-out games. The previous implementation is preserved under
> the `legacy-v0` tag; [docs/AUDIT.md](docs/AUDIT.md) explains why it was replaced.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/), which installs the pinned Python (3.13) if needed.

```bash
git clone https://github.com/NikaAtanelishvili/chess-ai && cd chess-ai
uv sync                                  # create .venv/ with the dependencies
uv run chessnet download-weights         # the released network, SHA-256 checked
uv run chessnet play                     # play it in the terminal, as White
```

`download-weights` saves `base-best.pt` where the engine looks for it,
`checkpoints/session2/base-best.pt` (42 MB; `--onnx` also fetches the ONNX export). The
files are attached to the [v1.0.0 release](https://github.com/NikaAtanelishvili/chess-ai/releases/tag/v1.0.0)
with their SHA-256. To use another network, pass `--weights`, set the `Weights` UCI option,
or set `CHESSNET_WEIGHTS`.

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
laptop CPU it searches about 400 to 500 playouts per second, so the default 800 playouts
take 1.5 to 2 s per move; 1,600 take about 4 s.

**En Croissant (Ubuntu):**

1. Install En Croissant from its [releases page](https://github.com/franciscoBSalgueiro/en-croissant/releases)
   (`.deb` or AppImage).
2. Open **Engines** (bottom of the left sidebar), click **Add New** (in some versions a
   **+** button), choose the **Local** tab, click **Binary file**, and select
   the `chessnet-uci` path printed above. Save the engine, which is named
   `chessnet 1.0.0`.
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
other moves. Level or worse, a draw is welcome.

The engine understands `go nodes`, `movetime`, `depth`, `wtime`/`btime`/`winc`/`binc`/
`movestogo`, and `infinite` with `stop`. MCTS has no fixed depth, so `go depth N` runs
`10 x N^2` playouts (at most a million): 10 at depth 1, 810 at depth 9, 4,000 at depth 20.
With a clock it thinks for `remaining / 30 + 0.8 x increment`,
less the move overhead, and never more than half the remaining time. Scores are converted
from the network's expected outcome `v` with Lichess's win-chance model,
`cp = ln((1 + v) / (1 - v)) / 0.00368208` (`v = 0.18` is about one pawn).

## Results

Measured with `chessnet eval` and `chessnet ladder` on `base-best.pt`; every game is in
`eval/results/`, and [docs/MODEL_CARD.md](docs/MODEL_CARD.md) has the full tables, the
intervals and the interpretation.

| Test | Result |
|------|--------|
| Policy-safe vs a random mover, 100 games | 100% (the raw policy: 97.5%, 5 draws, no losses) |
| MCTS 400 vs the raw policy, 100 games | 100% |
| MCTS 800 vs Stockfish at its weakest (UCI_Elo 1320 = Skill Level 0), 100 games | 100% |
| MCTS 800 on Stockfish's limited-strength UCI_Elo scale | 2493 (95% CI 2402 to 2595) |
| Strength against playouts, 50 to 1,600 | +94 Elo per doubling (95% CI +62 to +129), no flattening |
| Lichess puzzles, first move, by rating up to 1000 / 1500 / 2000 / above | policy 83 / 69 / 56 / 49%; MCTS 800 99 / 98 / 92 / 85% |
| `wide` (7.6M parameters) vs `base`, equal playouts / 1 s per move (T4) / 1 s per move (laptop CPU) | 49% / 55% / 5% |

**The rating is not a human rating.** It is on Stockfish's limited-strength UCI_Elo scale,
which is not comparable with FIDE or Lichess ratings. Limited-strength Stockfish weakens
itself by injecting mistakes, which a search engine exploits easily, and its levels
disagree (85% at UCI_Elo 2100 implies about 2400, 84% at 2300 about 2590). A rating earned
by a Lichess bot account would be the real-world check.

## Limitations

- The network imitates players rated 2000 and up: the policy proposes human moves, and the
  value estimates how humans fare from a position, not a theoretical evaluation.
- Tactics come from the search, which is written in Python: about 400 to 500 playouts per
  second on a laptop CPU, about 1,300 on a T4. Strength grows by roughly 94 Elo per
  doubling of the playouts.
- No opening book, endgame tablebases or pondering; standard chess only (no Chess960).
- Positions unlike human games, such as odd starting setups, are judged poorly.

## Evaluate

```bash
uv run chessnet eval --suite quick              # about 10 minutes on a laptop CPU
uv run chessnet eval --suite full --estimate    # times a few games, estimates the full suite
uv run chessnet eval --suite full --workers 2 --compare checkpoints/session2/wide-best.pt \
    --puzzles data/puzzles/lichess_db_puzzle.csv.zst
uv run chessnet ladder --workers 2              # Stockfish UCI_Elo ladder and playout scaling
```

The quick suite plays the raw policy and policy-safe against a random mover and MCTS (400
playouts) against the policy, and scores 500 Lichess puzzles. The full suite plays more
games, adds MCTS (800 playouts) against Stockfish at its lowest `UCI_Elo` and at `Skill
Level` 0, optionally a second network against the first, and 2,000 puzzles. Games start
from 50 balanced openings, each played with both colours. Results go to `eval/results/`
(JSON and PGNs), and a summary table to [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

- **Stockfish** is optional; its matches are skipped without it. On Ubuntu:
  `sudo apt install stockfish`.
- **Puzzles:** 500 are committed in `tests/fixtures/puzzles/`. The full Lichess puzzle
  database (CC0, 310 MB) is not:
  `curl -L -o data/puzzles/lichess_db_puzzle.csv.zst https://database.lichess.org/lichess_db_puzzle.csv.zst`
- Long runs belong on Kaggle: the kernel `kaggle/eval/` runs suites or ladders on the T4 x2
  machine (set `CHESSNET_REF`, `TASK` and `MAX_WALL_TIME` in `kaggle/eval/eval.py`, then
  `uv run kaggle kernels push -p kaggle/eval`).
- Long runs cannot hang: a stuck game or engine is killed and counted as an error, a
  heartbeat line appears every minute, each match is saved as soon as it ends, and
  `--max-wall-time 6h` stops cleanly with what has finished. `--only <part>,...` runs
  single matches or puzzle runs and merges them into earlier results; `--merge-results`
  merges a results file from another machine.

## Reproduce the training

The data and training run on Kaggle: a CPU kernel turns a monthly Lichess dump into
shards, published as a Kaggle Dataset, and a GPU kernel trains on them (10.1 h on one T4
for the released network). [docs/TRAINING.md](docs/TRAINING.md) has the data recipe, the
kernels and their pinned commits, the hyper-parameters, the measured throughput and the
GPU hours used. The Kaggle CLI (`uv run kaggle ...`) authenticates with an API token from
<https://www.kaggle.com/settings/api>, saved to `~/.kaggle/access_token` or exported as
`KAGGLE_API_TOKEN`.

## Development

```bash
uv sync --all-groups --all-extras         # every dependency, including dev tools
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

| Task | Command |
|------|---------|
| Lint and format | `uv run ruff check . && uv run ruff format .` |
| Type check | `uv run mypy` |
| Fast tests | `uv run pytest -m "not slow"` |
| All tests | `uv run pytest` |

PyTorch is installed from the CPU wheel index by default. Dependencies are declared in
`pyproject.toml` and pinned in `uv.lock`; add them with `uv add` (or `uv add --group dev`).
The design is described in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and decisions are
logged in [docs/DECISIONS.md](docs/DECISIONS.md).

Commits follow [Conventional Commits](https://www.conventionalcommits.org/) using the types
`feat`, `fix`, `refactor`, `perf`, `style`, `test`, `docs`, `build`, `ops` and `chore`
(enforced by a commit-msg hook).

## License

[MIT](LICENSE). The training data and puzzles come from the Lichess open database (CC0).
