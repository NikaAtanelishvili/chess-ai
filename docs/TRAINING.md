# Training data and runs

This document covers how the training data is built. Training runs and the Kaggle training
playbook will be added with the training loop.

## Data recipe

**Source.** The [Lichess open database](https://database.lichess.org/) monthly dumps of
standard rated games (CC0), e.g.
`https://database.lichess.org/standard/lichess_db_standard_rated_2026-10.pgn.zst` (31 GB
compressed). The dump is streamed: it is never stored or decompressed on disk, and a run
stops as soon as it has kept enough games.

**Filters** (`configs/data/lichess_2000plus.yaml`). A game is kept if:

| Check | Kept when |
|-------|-----------|
| Event | contains `Rated` |
| Variant | standard chess from the normal start (no `Variant`, `FEN` or `SetUp` header) |
| Result | `1-0`, `0-1` or `1/2-1/2` |
| Termination | `Normal` or `Time forfeit` (abandoned games and rules infractions are dropped) |
| Time control | base time of at least 180 s (no bullet, no correspondence) |
| Ratings | both players rated 2000 or more |
| Length | at least 20 plies |

All but the length check run on the raw header bytes before any parsing. The manifest
counts rejected games per reason.

**Records.** Each kept game contributes one 106-byte record per position: the position
before the move as the side to move sees it, the move played from it, the game result
from the mover's perspective (win/draw/loss), the ply and the mover's rating. The layout
is in `docs/ARCHITECTURE.md`.

**Splits.** By game: BLAKE2b of the game URL, mod 100. Values 0-97 go to train, 98 to val,
99 to test.

**Shuffling.** Positions are scattered into random bucket files while games stream in.
Each bucket is then permuted and cut into shards of 1,000,000 positions
(`train-00000.npy`, ...) and deleted. A shard is therefore a random sample of its split,
and peak disk use stays near one dataset plus one bucket.

**Output.**

```
shards/
├── manifest.json        # encoding version, config, source, counts, SHA-256 per shard, git commit
├── train-00000.npy ...
├── val-00000.npy
└── test-00000.npy
```

`manifest.json` is written last; a directory without it is an incomplete run. Loading data
built with a different `ENCODING_VERSION` fails.

## Building the dataset

### Locally

```bash
uv run chessnet preprocess \
    --config configs/data/lichess_2000plus.yaml \
    --source https://database.lichess.org/standard/lichess_db_standard_rated_2026-10.pgn.zst \
    --out data/shards
```

- `--max-games N` overrides the config's game count.
- `--workers N` sets the number of parser processes (default: one per CPU core).
- `--source` also accepts a local `.pgn.zst` or `.pgn` file.

A run against the test fixture takes about a second:

```bash
uv run chessnet preprocess --config configs/data/lichess_2000plus.yaml \
    --source tests/fixtures/mini.pgn.zst --out /tmp/shards
```

### On Kaggle (CPU kernel)

`kaggle/preprocess/` is a script kernel: CPU only, internet on, so it does not use GPU
quota (sessions are still capped at 12 h).

1. In `kaggle/preprocess/preprocess.py`, set `CHESSNET_REF` to a tag or commit SHA. In
   `kernel-metadata.json`, replace `<kaggle-username>`.
2. Push and start it: `uv run kaggle kernels push -p kaggle/preprocess`.
3. Follow it with `uv run kaggle kernels status <kaggle-username>/chessnet-preprocess`, or
   watch the live log on the kernel's page.
4. When it finishes, `/kaggle/working/shards` is the kernel's output. Turn it into the
   dataset `chessnet-lichess-2000plus-v1`:
   - **Kaggle UI (no download):** open the kernel → *Output* → *New Dataset*. Name it
     `chessnet-lichess-2000plus-v1` and keep it private.
   - **CLI:** `uv run kaggle kernels output <kaggle-username>/chessnet-preprocess -p ./shards`
     downloads the output (about 8 GB). Then run `uv run kaggle datasets init -p ./shards`,
     set the title and id in `dataset-metadata.json`, and run
     `uv run kaggle datasets create -p ./shards`. Later versions use `kaggle datasets
     version -p ./shards -m "<message>"`.

The kernel's working directory holds at most 20 GB. A run at the default size needs about
8 GB of shards plus one bucket of temporary space.

## Measured throughput

Measured on 2026-11-25 against the October 2026 dump with `--max-games 20000`, on the
development machine (4-core laptop CPU, home internet connection).

| Run | Read | Games scanned/s | Games kept/s | Kept fraction | Wall time |
|-----|------|-----------------|--------------|---------------|-----------|
| Real URL, 4 workers (network-bound) | 0.93 MB/s compressed (6.7 MB/s decompressed) | 2,860 | 176 | 6.2% | 114 s for 20,000 games |
| Local copy of the first 300,000 games, 4 workers (CPU-bound) | 19.7 MB/s decompressed | 8,450 | 533 | 6.3% | 36.5 s |

From the 20,000-game run:

- Each kept game has 79.6 positions on average; 1,592,852 positions took 162 MB of shards.
- Kept games cost 5.3 KB of compressed download each: 325k games scanned for 106 MB read.
- Rejections: rating 55.5%, time control 37.4%, rated check 0.4%, termination 0.3%,
  result < 0.1%; 259 games were too short.
- Val and test received 206 and 182 games (about 1% each, as intended).
- No move was unrepresentable.
- The main process peaked at 320 MB resident memory.

The real-URL run was limited by this machine's connection to `database.lichess.org`
(about 1 MB/s). The CPU-bound run measures the pipeline itself: there, the parser workers
are the bottleneck (4 workers beat 3 by about 12%).

### Estimate for the default dataset (1,000,000 games, ~80M positions, ~8.4 GB)

- **Download:** about 5.3 GB of the 31 GB dump (~16M games scanned).
- **CPU:** about 31 min at the measured 533 kept games/s. Kaggle's 4 vCPUs are slower than
  this laptop, so expect roughly 50-80 min.
- **Network:** under 10 min at 10 MB/s or more. At this laptop's 1 MB/s it would take about
  1.5 h, and the run would be network-bound.

**Expected wall time on a Kaggle CPU kernel: about 1-2 h, at most around 3 h if the
download is slow.** That is well inside the 12-hour session limit. Peak disk use is about
8.5 GB, within Kaggle's 20 GB.
