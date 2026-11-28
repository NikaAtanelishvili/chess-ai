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
  `kaggle_t4.yaml` (the base model, 5 epochs at batch 2048). Kaggle has retired its P100,
  so there is no P100 config.
- **Precision:** fp16 autocast with a GradScaler on CUDA; fp32 on the CPU.
- **Speed options:** `compile: true` and `channels_last: false` are the defaults, the
  fastest combination on a T4 (see the session 1 results below). Validation always runs
  uncompiled. Turn compilation off for short CPU runs, where it only adds start-up time.
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
- machine `NvidiaTeslaT4`, two T4s: GPU 0 for the main run, GPU 1 for a second run;
- internet on;
- input: the dataset `nikaatanelishvili/chessnet-lichess-2000plus-v1`.

It installs the commit pinned in `CHESSNET_REF` and runs `pytest -m gpu --require-gpu`,
which fails if no GPU is attached or the GPU unpacking path is wrong. It then finds the
dataset by globbing `/kaggle/input` for `manifest.json`, and stops with a clear error if it
finds none or more than one.

### Session 1 results (2026-11-25)

Session 1 ran for 7.4 min on the T4 x2 machine: GPU tests (1 passed), the benchmark
(5.7 min) and the smoke run (1 min). Training used one of the two T4s.

Training speed on one T4 (positions/s; torch 2.11.0+cu130, fp16 mixed precision):

| Model | Batch | Plain | channels_last | compile | Both |
|-------|------:|------:|--------------:|--------:|-----:|
| small | 1024 | 12,525 | 15,891 | **17,668** | 16,970 |
| small | 2048 | 12,000 | 14,686 | 16,783 | 15,893 |
| base | 1024 | 6,547 | 8,827 | 10,058 | 9,615 |
| base | 2048 | 6,341 | 9,057 | **10,639** | 9,712 |

- `torch.compile` alone is fastest for both models: +40% (small) and +54-68% (base) over
  plain eager. channels_last alone helps too (+22-43%), but adding it to a compiled model
  makes every configuration slower.
- Compilation costs 17-25 s on the first step of every session.
- Forward passes alone (validation, which runs uncompiled): 20,514 positions/s for base at
  batch 2048.
- Peak GPU memory for base at batch 2048, compiled: 1.71 GB of 15 GB.
- Data loader with 2 workers: 159,089 positions/s at batch 1024 and 359,914 at batch 2048,
  15 to 34 times what the GPU consumes.

Smoke run (tiny model, 1000 steps of 1024): about 30,000 positions/s in the training loop.
Validation top-1 rose from 23.8% to 31.1% (top-5 66.2%, WDL accuracy 52.9%). The
White/Black top-1 gap at the four validations was 0.94, 0.58, 0.56 and 0.18 points, within
the sampling noise of the 20,480-position subset (about +/-0.65 points).

### Numbers for planning

| Quantity | Value | Source |
|----------|-------|--------|
| Training positions | 77,107,653 | dataset manifest |
| Steps per epoch (batch 2048) | 37,572 | shard sizes; each shard drops its last partial batch |
| Base training speed (T4, fp16, batch 2048, compiled) | 10,639 positions/s | measured in session 1 |
| Time per epoch (base) | 2.01 h | measured speed |
| Validation (51,200 positions every 2500 steps) | about 3.5 s each, with `best.pt`; 76 per run | measured forward speed |
| Checkpoint size (base, with optimizer state) | ~42 MB; 3 rolling + best kept | 3.5M parameters |
| 5 epochs (`kaggle_t4.yaml`) | 10.18 h, 187,860 steps | measured speed; see below |

Each shuffled epoch starts every shard at a random offset, so the epochs after the first can
be a few dozen steps shorter (at most one per shard). The 10.18 h are 10.05 h of training
plus 8 min for validation, checkpoints (every 20 min and at each epoch end), start-up and
compilation. `chessnet.train.budget` computes the estimate, and a test pins these figures.

**Epochs** is the largest whole number whose estimated run time leaves at least 1 h of
`--max-wall-time 11h15m`: 5 epochs take 10.18 h and leave 1.07 h, while 6 would take
12.2 h. The run still finishes inside the session if its sustained speed is up to 9.6%
below the benchmark's. If it is slower, the run stops cleanly at 11 h 15 min, and session 3
finishes it. The learning-rate schedule spans the whole planned run and is fixed when the
run starts.

### The second run (GPU 1)

The main run uses one of the machine's two T4s. On the other, the kernel trains a second,
independent run: the `wide` model (10 blocks x 192 filters, 7,572,332 parameters, 2.1 times
base) with the main run's settings (batch 2048, `torch.compile` on, channels_last off, the
same schedule shape and validation subset).

- **Purpose:** an equal-time comparison. Is a bigger model trained for fewer epochs (wide)
  better than base trained for 5 epochs, on the same data and the same GPU time?
- **Epochs** are chosen during the session. Once the main run is training, a benchmark of
  `wide` on GPU 1 (about 2 minutes) measures its speed. The kernel then takes the largest
  whole number of epochs whose estimate (computed as above) leaves at least 1.5 h of the
  second run's wall time, which ends with the main run's: 11 h 15 min after the main run
  started. If not even one epoch fits, the second run is skipped. `run-wide/epochs.json`
  records the measured speed, the estimate and the epochs, or why the run was skipped;
  `run-wide/benchmark.md` holds the benchmark.
- **Isolation:** the main run starts first, and nothing about the second run can stop, slow
  or end it.
  - The runs use separate processes, GPUs (`CUDA_VISIBLE_DEVICES`), output directories
    (`run-base/`, `run-wide/`) and logs: the main run's goes to the kernel log, the second
    run's to `run-wide/train.log`.
  - The second run starts only after the main run has logged training progress, so their
    compilations never overlap. It runs at the lowest CPU priority (`nice -n 19`), with one
    compiler thread, one OpenMP thread and its own compiler caches.
  - Each run has one DataLoader worker instead of two (the machine has 4 vCPUs). One worker
    delivers about 180,000 positions/s, 17 times what base consumes.
  - If the second run fails, the kernel logs it and keeps waiting for the main run. The
    kernel exits only when both runs have ended, with the main run's exit code, and it never
    signals the main run. A second run still going 10 minutes after its deadline gets
    SIGTERM (it saves a checkpoint and stops), then SIGKILL 10 minutes later, so the session
    ends well before Kaggle's 12-hour limit. If the main run fails, the second run carries
    on; cancel the session on the kernel page to fix and restart sooner.
  - On a machine with one GPU, only the main run trains.
- **Disk:** a checkpoint with optimizer state takes 42.5 MB for base and 91.1 MB for wide.
  Each run keeps 3 rolling checkpoints and `best.pt`, so the two runs write less than
  0.6 GB of Kaggle's 20 GB output limit. The kernel logs the output size at the end.

### Sessions (one week's GPU quota, about 30 h)

| Session | `MODE` | Purpose | GPU hours |
|---------|--------|---------|-----------|
| 0 | (CPU kernel) | Preprocessing: done, 0.92 h, no GPU quota | 0 |
| 1 | `smoke` | Done: GPU tests, benchmark (16 configurations), tiny model on 1M positions | 0.12 |
| 2 | `train` | Main run (base, 5 epochs, about 10.2 h) and the second run (wide); both stop by 11 h 15 min | <= 11.6 |
| 3 | `train` | Only if a run of session 2 stops before its end: each run resumes from its own directory | 0-2 |
| - | - | Reserve for a fix or a re-run | ~5 |

### GPU hours used

Kaggle counts a GPU session's whole length against the weekly quota. Every GPU session of
this project so far, from the kernel logs:

| Date | Kernel | Machine | What | GPU hours |
|------|--------|---------|------|----------:|
| 2026-11-24 | `chessnet-preprocess` | CPU | the dataset (0.92 h, no GPU quota) | 0 |
| 2026-11-25 | `chessnet-train`, session 1 | T4 x2 | GPU tests, benchmark, smoke run | 0.12 |
| 2026-11-25 | `chessnet-train`, session 2 | T4 x2 | base (10.08 h) and wide (8.73 h) at once | 10.10 |
| 2026-11-26 | `chessnet-search` | T4 | search speed | 0.04 |
| 2026-11-27 | `chessnet-eval`, full suite | T4 x2 | three matches in 20 min, then hung until stopped | about 8.8 |
| 2026-11-27 | `chessnet-eval`, the missing parts | T4 x2 | Skill Level 0, wide vs base, puzzles | 1.65 |
| 2026-11-27 | `chessnet-eval`, ladder | T4 x2 | UCI_Elo ladder and playout scaling | 1.81 |
| | | | **Total** | **about 22.5** |

Training the released network took 10.1 GPU hours. The 8.5 hours lost to the hung
evaluation led to the supervised workers (DECISIONS.md, "Hang-proof evaluation runs").

### Steps

1. **Session 1 (smoke)**, done on 2026-11-25 (results above). In `kaggle/train/train.py`,
   `MODE = "smoke"` and `CHESSNET_REF` set to the pinned commit. Then:

   ```bash
   uv run kaggle kernels push -p kaggle/train
   uv run kaggle kernels status nikaatanelishvili/chessnet-train
   uv run kaggle kernels output nikaatanelishvili/chessnet-train -p runs/smoke \
       --file-pattern '.*(benchmark\.md|metrics\.csv|runs\.jsonl)$'
   ```

   The file pattern skips checkpoints and TensorBoard files; the kernel log
   (`chessnet-train.log`, JSON lines) is downloaded too. Check:
   - the GPU tests passed (in the log);
   - `benchmark.md` gives positions/s for small and base at batch 1024 and 2048, each with
     `torch.compile` and channels_last off and on, and the data loader's speed;
   - `smoke/val_metrics.csv` shows top-1 rising well above chance, and `top1_colour_gap`
     staying within a few points.

2. **Before session 2**, done. The measured numbers are in this playbook, and
   `configs/train/kaggle_t4.yaml` has `epochs`, `batch_size`, `compile` and
   `channels_last` set from them. `CHESSNET_REF` points to the commit that contains those
   settings, and `MODE = "train"`.

3. **Session 2.** Push again. The main run's first progress line should appear about 2
   minutes after training starts and show about 10,600 positions/s. A few minutes later the
   kernel logs the wide model's measured speed, its epochs, and the start of the second run.
   The main run finishes after about 10.2 h (11 h 15 min at most) with checkpoints in
   `/kaggle/working/run-base`; the second run writes to `/kaggle/working/run-wide`. Follow
   the main run in the live log (a line every 200 steps, with an ETA). The second run's log
   is `run-wide/train.log` in the output. `tensorboard --logdir runs/<name>` shows both runs.

4. **Continuing (session 3).** If either run is not finished, turn session 2's output into
   the dataset `nikaatanelishvili/chessnet-run`: kernel page, *Output*, *New Dataset* (or
   *New Version* later). Add it to `dataset_sources` in `kaggle/train/kernel-metadata.json`
   and push again. The kernel copies `run-base/` and `run-wide/`, and each run continues
   from the newest checkpoint in its own directory, at the same step and position in the
   data; a finished run ends at once. The second run keeps the epochs chosen in session 2
   (`run-wide/config.yaml`).

5. **After the run.** Download the output (`kaggle kernels output ...`). Each run's
   `best.pt` and last checkpoint are the candidates for evaluation, which also settles the
   equal-time comparison of wide and base.

## Session 2 results (2026-11-25)

Both runs of session 2 (code `fd50dcf`) finished all their planned epochs, well inside the
wall-time budget, and wrote 0.54 GB of output. The GPU tests passed first.

| | base (main run, GPU 0) | wide (second run, GPU 1) |
|---|---:|---:|
| Model | 10 x 128, 3,526,092 parameters | 10 x 192, 7,572,332 parameters |
| Epochs, steps | 5, 187,860 (completed) | 2, 75,144 (completed) |
| Run time | 10.08 h (estimated 10.18 h) | 8.73 h (estimated 7.83 h) |
| Mean training speed | 10,621 positions/s (benchmark 10,639) | 4,904 positions/s (benchmark 5,533) |
| Val top-1 / top-3 / top-5 | **54.75%** / 82.96% / 91.56% | **54.99%** / 83.20% / 91.75% |
| Val WDL accuracy | 57.69% | 57.69% |
| Value loss, val / train | 0.848 / 0.828 | 0.848 / 0.830 |
| Policy loss, val / train | 1.910 / 1.881 | 1.907 / 1.888 |
| Top-1, White / Black to move | 54.30% / 55.21% (gap 0.91 points) | 54.70% / 55.27% (gap 0.57 points) |
| Top-1, opening / middlegame / endgame | 54.9% / 53.4% / 56.8% | 54.8% / 53.8% / 57.1% |

Validation is on the fixed subset of 51,200 positions from held-out games. Train figures
are means over the last 2% of logged steps. Both runs pass the minimum target for
validation top-1 (40%) and the stretch target (50%).

- **No overfitting.** Validation value loss fell through each run and was at its lowest at
  the end (or within 0.0001 of it). The train-validation gap is 0.02 for the value loss and
  0.03 for the policy loss, so the 77M training positions are far from memorised.
- **White/Black gap.** Below 1 point at the end of both runs. Across all validations, base
  peaked at 1.28 points (8 of 76 validations above 1 point) and wide at 1.06 points.
  Black-to-move positions score slightly higher in both runs. A position and its colour
  mirror encode to identical planes, so the network cannot treat the colours differently.
  The gap therefore comes from the positions in the subset, not from the encoding.
- **Speed.** The main run held its benchmark speed. The second run averaged 11% below its
  2-minute benchmark, so it took 8.73 h instead of 7.83 h. Its 1.5 h margin covered that,
  and it finished 2.5 h before its deadline.

### Equal-time comparison

The wide model trained for 2 epochs reached the same quality as base trained for 5, on 13%
less GPU time: +0.24 points of top-1, +0.19 points of top-5, and equal value metrics. The
difference is within the noise of a 51,200-position subset (about 0.2 points). Base gained
1.3 points of top-1 over its last three epochs and 0.5 over its last two, while its learning
rate annealed, so extra passes over the same data return little.

- **Network for search and play: base** (`base-best.pt` below). It matches wide's accuracy
  and needs about half the computation per evaluation, so it searches about twice as many
  nodes in the same time. That matters most on a CPU.
- **For a future, bigger run:** at equal GPU time a 2.1x wider model with 2 epochs matched
  base with 5, so further compute should go into model size and more games, not more epochs.
  With DDP across both T4s (about 1.8x; see the backlog in `docs/DECISIONS.md`), one session
  could train a wide-sized or bigger model for 3 to 4 epochs. More training data (several
  months of games instead of one) would raise the ceiling further.

### Checkpoints

Stored outside git in `checkpoints/session2/` (ignored by git). Source: the session 2 output
of the Kaggle kernel `nikaatanelishvili/chessnet-train`, downloaded with:

```bash
uv run kaggle kernels output nikaatanelishvili/chessnet-train -p runs/session2 \
    --file-pattern '.*(best\.pt|step-000187860\.pt|step-000075144\.pt|metrics\.csv|runs\.jsonl|epochs\.json|train\.log|benchmark\.(md|json|log)|config\.yaml)$'
```

Every checkpoint has encoding version 1 and dataset manifest hash `88d5cddd...6793ea`.

| File | Origin | Step | Val top-1 | SHA-256 |
|------|--------|-----:|----------:|---------|
| `base-best.pt` | `run-base/best.pt` (lowest val loss) | 182,500 | 54.73% | `370248af31086c8de72b9df0ef03a41a426e446a1fe31d6714fd0fe2306deb78` |
| `base-step-000187860.pt` | `run-base`, final | 187,860 | 54.75% | `d6cb43678d382ad9583be5515b3404bba2ffedca442dfa79dd2b166f6f19137b` |
| `wide-best.pt` | `run-wide/best.pt` (lowest val loss) | 75,000 | 54.94% | `7a3d263826103e6f85fa74c55f95b21ab988a8e7680b8eb1be94ac4e1cd4ca74` |
| `wide-step-000075144.pt` | `run-wide`, final | 75,144 | 54.99% | `a5e77a70c3a58b4e1bb0e12eba37e5157e4db05728f9cd4e09167c25cf890cc6` |

Check a copy with `sha256sum checkpoints/session2/*.pt`.

### Sanity check of base-best.pt (CPU, policy only)

Top-5 moves from the masked policy (softmax over legal moves):

| Position | Top-5 |
|----------|-------|
| Start position (White to move) | e4 52.1%, d4 30.1%, Nf3 6.6%, c4 5.2%, b3 1.3% |
| 1.e4 (Black to move) | c5 31.2%, e5 22.4%, e6 12.8%, c6 12.2%, d5 6.8% |
| Ruy Lopez, 1.e4 e5 2.Nf3 Nc6 3.Bb5 (Black to move) | a6 48.7%, Nf6 22.7%, Bc5 8.6%, f5 5.8%, d6 5.6% |
| Queen's Gambit, 1.d4 d5 2.c4 (Black to move) | e6 41.9%, c6 30.9%, dxc4 11.7%, Nf6 6.5%, e5 4.4% |
| 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6?? (White to move, mate in 1) | Qxf7# 90.4%, Bxf7+ 7.5%, Qh4 1.0%, Qf3 0.4%, Qe2 0.2% |

The value head rates the start position about even (+0.04 for White). After 3...Nf6?? it
gives White only +0.29, because it does not calculate the mate; the search will.

## Reproducing the released network

The release `v1.0.0` ships `base-best.pt`, its ONNX export, the dataset's `manifest.json`
and their SHA-256. Every step that produced the network is pinned:

1. **Data:** the preprocessing kernel at `v0.1.0` (`873a1af`) streams the October 2026
   Lichess dump (`lichess_db_standard_rated_2026-10.pgn.zst`) into shards with the
   filters and seed recorded in `manifest.json`. The Kaggle Dataset built that way is
   private, so to reproduce it, run the kernel (or `chessnet preprocess` locally) and
   compare each shard with the SHA-256 in `manifest.json`. The manifest's own SHA-256,
   `88d5cddd...6793ea`, is stored in the checkpoint.
2. **Training:** the training kernel `kaggle/train/` pinned to `fd50dcf` with `MODE =
   "train"` and `configs/train/kaggle_t4.yaml` (session 2 above). The checkpoint records
   the commit (`git_sha`), the config and the dataset hash; `base-best.pt` is the step with
   the lowest validation loss.
3. **Evaluation:** `kaggle/eval/` and `chessnet eval` / `chessnet ladder`, with every
   game in `eval/results/`.

Training on a GPU uses fp16 and `torch.compile`, so a rerun matches in its metrics, not
bit for bit; on the CPU, training is deterministic (DECISIONS.md, "Deterministic training
on the CPU").
