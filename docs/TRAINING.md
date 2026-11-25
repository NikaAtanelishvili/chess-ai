# Training data and runs

How the training data is built, how training runs, and the playbook for running it on
Kaggle.

## Data recipe

**Source.** The [Lichess open database](https://database.lichess.org/) monthly dumps of
standard rated games (CC0), e.g.
`https://database.lichess.org/standard/lichess_db_standard_rated_2026-10.pgn.zst` (31 GB
compressed). The dump is streamed: it is never stored or decompressed on disk, and a run
stops as soon as it has kept enough games.

**Filters** (`configs/data/lichess_2000plus.yaml`). A game is kept if:

| Check | Kept when |
|-------|-----------|
| Rated | assumed: the source must be a rated-games dump. The Event header is not checked, so rated Swiss games (`Blitz swiss ...`) are kept. |
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

### Kaggle CLI setup

The Kaggle CLI (pinned in the `dev` group, `uv run kaggle ...`) authenticates with an API
token:

1. On <https://www.kaggle.com/settings/api>, click *Generate New Token* under *API* and copy
   the token.
2. Give it to the CLI in one of two ways:
   - **Token file (recommended):** save it to `~/.kaggle/access_token`
     (`%USERPROFILE%\.kaggle\access_token` on Windows), readable only by you:
     `chmod 600 ~/.kaggle/access_token`.
   - **Environment variable:** `export KAGGLE_API_TOKEN=<token>`. It may also hold the
     path of a file that contains the token.
3. Check it: `uv run kaggle kernels list --mine`.

The CLI looks for `KAGGLE_API_TOKEN` first, then `~/.kaggle/access_token`. The legacy
`~/.kaggle/kaggle.json` key and OAuth (`uv run kaggle auth login`) still work, but are not
needed. Keep tokens outside the repository; `.gitignore` excludes `kaggle.json` and
`access_token` files in case one is copied in.

### On Kaggle (CPU kernel)

`kaggle/preprocess/` is a script kernel: CPU only, internet on, so it does not use GPU
quota (sessions are still capped at 12 h).

1. `kaggle/preprocess/preprocess.py` pins `CHESSNET_REF` to the commit it installs (a tag or
   commit SHA, never a branch), and `kernel-metadata.json` names the kernel
   `<kaggle-username>/chessnet-preprocess`. Update both when pinning a new version or using
   another account.
2. Push and start it from the repository root: `uv run kaggle kernels push -p kaggle/preprocess`.
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
- Rejections: rating 55.5%, time control 37.4%, termination 0.3%, result < 0.1%; 259
  games were too short. This run still checked Event for `Rated`, which rejected 0.4%
  (rated Swiss games). That check has since been dropped, which keeps about 0.2-0.3% more
  games.
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

## The dataset built on Kaggle (v1)

Built by the Kaggle CPU kernel at v0.1.0 (`873a1af`) and published as the private dataset
`nikaatanelishvili/chessnet-lichess-2000plus-v1` (8.34 GB, 81 files):

| | |
|---|---|
| Source | October 2026 dump; 5.18 GB read (36.9 GB decompressed), stopped at 1,000,000 kept games |
| Kernel time | 0.92 h, faster than the estimate above |
| Games | 15.8M scanned, 1,014,144 passed the header filter, 1,000,000 kept (6.33%) |
| Rejected | rating 9.01M, time control 5.73M, termination 45k, too short 13.7k, result 1.7k |
| Train | 979,935 games, 77,107,653 positions, 78 shards (77 x 1M + 107,653) |
| Val | 9,983 games, 785,383 positions, 1 shard |
| Test | 10,082 games, 799,013 positions, 1 shard |

Encoding version 1, no skipped positions, a SHA-256 per shard in the manifest.

## Training

```bash
uv run chessnet train \
    --config configs/train/kaggle_t4.yaml \
    --data /path/to/shards \
    --out runs/base-1 \
    --max-wall-time 11h15m
```

- **Configs** (`configs/train/`): `smoke.yaml` (the tiny model, 1000 steps) and
  `kaggle_t4.yaml` (the base model, 3 epochs, batch 1024). Kaggle has retired its P100, so
  there is no P100 config.
- **Precision:** fp16 autocast with a GradScaler on CUDA; fp32 on the CPU.
- **Optimisation:** AdamW (SGD with Nesterov momentum is an option); linear warmup, then
  cosine decay over the whole planned run; gradient clipping at norm 5.
- **Loss:** policy cross-entropy with label smoothing 0.05, plus 0.5 x the WDL value
  cross-entropy. There is no legal-move mask during training.
- **Validation:** every `val_every` steps on a fixed subset (the first `val_batches`
  batches of the val split).
- **Checkpoints** in `<out>/checkpoints/`: one every 20 minutes and one at the end of each
  epoch; the last 3 are kept. `<out>/best.pt` holds the best validation loss.
- **Resume:** run the same command again with the same `--out`, or add `--resume-from <dir>`
  to start from another run's checkpoints. Everything is restored, including the position
  in the epoch, and an interrupted run continues exactly as an uninterrupted one would
  (a test compares both, step by step). A checkpoint from another dataset or another model
  is refused.
- **Stopping:** SIGTERM, SIGINT and `--max-wall-time` stop after the current step and write
  a checkpoint.
- **Logs** in `<out>/`:
  - a progress line every `log_every` steps on the console;
  - `metrics.csv` and `val_metrics.csv`;
  - TensorBoard events in `tb/` (`tensorboard --logdir <out>/tb`);
  - `runs.jsonl`, one line per session, with the config, git commit and dataset hash.

  `--logger wandb` adds Weights & Biases. It needs `WANDB_API_KEY`, and the run continues
  without W&B if it is missing or unreachable.
- **Metrics:** loss (policy, value), top-1/3/5 move accuracy, WDL accuracy, value MSE,
  top-1 by game phase (ply < 20, 20-60, > 60), learning rate, gradient norm, loss scale,
  positions/s and GPU memory.
- **Colour check:** `top1_white` and `top1_black` (and `top1_colour_gap`) split top-1
  accuracy by side to move. With a correct encoding they agree to within about a point; a
  larger gap means something is wrong with mirroring.

`scripts/benchmark_train.py` measures training positions/s for each model and batch size
with every combination of the two speed options, `torch.compile` and channels_last (config
flags `compile` and `channels_last`; they change speed only, never the checkpoints). It
also times forward passes alone, which is what validation costs. With `--data` it measures
the data loader too and reports hours per epoch.

## Kaggle run playbook

The training kernel `kaggle/train/` is a GPU script kernel:
- machine `NvidiaTeslaT4`, using one of the two T4s;
- internet on;
- input: the dataset `nikaatanelishvili/chessnet-lichess-2000plus-v1`.

It installs the commit pinned in `CHESSNET_REF` and runs `pytest -m gpu --require-gpu`,
which fails if no GPU is attached or the GPU unpacking path is wrong. It then finds the
dataset by globbing `/kaggle/input` for `manifest.json`, and stops with a clear error if it
finds none or more than one.

### Numbers for planning

| Quantity | Value | Source |
|----------|-------|--------|
| Training positions | 77,107,653 | dataset manifest |
| Steps per epoch (batch 1024) | 75,300 | manifest / 1024 |
| Base model training speed on a T4 (fp16) | ~10,000 positions/s (range 8,000-12,000) | **estimate**; session 1 measures it |
| Time per epoch (base) | ~2.1 h (1.8-2.7 h) | from the estimate |
| 3 epochs (`kaggle_t4.yaml`) | ~6.4 h, 225,900 steps | from the estimate |
| Data loader (2 workers) | >150,000 positions/s at batch 256 on a laptop CPU | measured locally; not a bottleneck |
| Checkpoint size (base, with optimizer state) | ~42 MB; 3 rolling + best kept | 3.5M parameters |
| GPU memory (base, batch 1024, fp16) | a few GB of the T4's 16 GB | estimate; session 1 measures it |

The speed estimate assumes about 1.2 GFLOP per training position for the base model, at
roughly a quarter of a T4's fp16 tensor-core peak. Replace it with the measured value from
session 1 (`benchmark.md`, column *Hours per epoch*).

### Sessions (one week's GPU quota, about 30 h)

| Session | `MODE` | Purpose | GPU hours |
|---------|--------|---------|-----------|
| 0 | (CPU kernel) | Preprocessing: done, 0.92 h, no GPU quota | 0 |
| 1 | `smoke` | GPU tests, throughput benchmark (16 configurations), tiny model on 1M positions | ~0.5 |
| 2 | `train` | Main run, part 1: base model, stops after 11 h 15 min | <= 11.5 |
| 3 | `train` | Main run, part 2, resumed from session 2 (only if needed) | 0-11.5 |
| - | - | Reserve for a fix or a re-run | ~5 |

At the estimated speed, 3 epochs fit into session 2 alone. With the measured speed `s`
(positions/s), the epochs that fit into one session are `11.25 h x 3600 x s / 77.1M`.
Session 1's measurement may justify `epochs: 4` (about 8.5 h at the estimate).
**Decide `epochs` before session 2:** the learning-rate schedule spans the whole planned
run and is fixed when the run starts.

### Steps

1. **Session 1 (smoke).** In `kaggle/train/train.py`, keep `MODE = "smoke"` and check that
   `CHESSNET_REF` is the pinned commit. Then:

   ```bash
   uv run kaggle kernels push -p kaggle/train
   uv run kaggle kernels status nikaatanelishvili/chessnet-train
   uv run kaggle kernels output nikaatanelishvili/chessnet-train -p runs/smoke
   ```

   Check:
   - the GPU tests passed (in the log);
   - `benchmark.md` gives positions/s for small and base at batch 1024 and 2048, each with
     `torch.compile` and channels_last off and on, and the data loader's speed;
   - `smoke/val_metrics.csv` shows top-1 rising well above chance, and `top1_colour_gap`
     staying within a few points.

2. **Before session 2.** Put the measured numbers into this playbook and set `epochs`,
   `batch_size`, `compile` and `channels_last` in `configs/train/kaggle_t4.yaml`. Pin
   `CHESSNET_REF` to the commit that contains those settings, and set `MODE = "train"`.

3. **Session 2 (main run).** Push again. The run stops after 11 h 15 min with a checkpoint in
   `/kaggle/working/run`. Follow progress in the live log (a line every 200 steps) or
   afterwards with `tensorboard --logdir runs/<name>/run/tb`.

4. **Continuing (session 3).** If the run is not finished, turn session 2's output into the
   dataset `nikaatanelishvili/chessnet-run`: kernel page, *Output*, *New Dataset* (or *New
   Version* later). Add it to `dataset_sources` in `kaggle/train/kernel-metadata.json` and
   push again. The kernel copies the previous `run/` directory and continues from its newest
   checkpoint, at the same step and position in the data.

5. **After the run.** Download the output (`kaggle kernels output ...`). `best.pt` and the
   last checkpoint are the candidates for evaluation.
