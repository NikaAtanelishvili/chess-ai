# Model card: chessnet base

The network that the engine plays with: `checkpoints/session2/base-best.pt` (not in git;
source and SHA-256 `370248af...` in [TRAINING.md](TRAINING.md#checkpoints)).

## Model

- **Architecture:** SE-ResNet with 10 residual blocks of 128 filters (squeeze-excitation
  ratio 4), a convolutional policy head with 4,672 move logits (73 move types x 64
  squares, AlphaZero's encoding) and a win/draw/loss value head; 3,526,092 parameters.
- **Input:** 19 planes of 8 x 8 from the side to move's perspective (pieces, castling,
  en passant, halfmove clock); `ENCODING_VERSION` 1. Details in
  [ARCHITECTURE.md](ARCHITECTURE.md).
- **Use in the engine:** priors for PUCT search and leaf values `P(win) - P(loss)`; or the
  policy alone (`Mode policy`).

## Training data

1,000,000 rated standard games from the Lichess open database (October 2026), both players
rated 2000 or more, base time of at least 3 minutes, ended normally or on time, at least 20
plies. Split by game: 77.1M training positions, 785k validation, 799k test. Each position
is labelled with the move played from it and the game's result for the side to move.
Recipe and figures: [TRAINING.md](TRAINING.md#data-recipe).

## Training

Supervised learning: policy cross-entropy (label smoothing 0.05) plus 0.5 x the WDL
cross-entropy, AdamW (learning rate 1e-3, warmup and cosine decay), batch 2048, 5 epochs,
fp16 on one Kaggle T4 (10.1 h). The checkpoint is the one with the lowest validation loss,
at step 182,500 of 187,860.

On 51,200 positions from held-out games:

| Metric | Value |
|--------|------:|
| Top-1 / top-3 / top-5 move accuracy | 54.7% / 83.0% / 91.6% |
| Top-1, White / Black to move | 54.3% / 55.2% |
| WDL accuracy | 57.7% (always "win": 45.4%) |
| Value loss (cross-entropy) | 0.848 (constant label frequencies: 0.961) |

The value head is calibrated: positions given a 0.8 to 0.9 chance of winning are won
83% of the time. It is most informative in endgames and close to a coin flip in openings,
where human games are.

## Intended use and limitations

- Playing and analysis through UCI (`chessnet-uci`) or `chessnet play`; research and
  teaching.
- It imitates 2000+ rated humans: the policy proposes human moves, and the value is the
  outcome humans reach from a position, not a theoretical evaluation.
- The network does not calculate. Tactics come from the search, which is slow in Python
  (about 500 nodes per second on a laptop CPU, 1,500 on a T4).
- Standard chess only (no Chess960). Positions unlike human games, such as a missing queen
  in the starting setup, are judged poorly.

## Evaluation

`chessnet eval --suite quick|full` writes `eval/results/<checkpoint>.<suite>.json`, PGNs
of every game, and the tables below. Matches start from 50 balanced openings, each played
once with each colour. Elo differences come with 95% confidence intervals. Puzzles are
scored on the first move of the solution, by Lichess rating.

<!-- eval: base-best quick -->
### base-best.pt, quick suite

2026-11-27, 11th Gen Intel(R) Core(TM) i3-1115G4 @ 3.00GHz (cpu, 1 worker(s)), chessnet 0.2.0.dev0. 50 balanced openings, each played with both colours.

| Match | Games | W / D / L | Score | Elo (95% CI) | Target |
|-------|------:|-----------|------:|--------------|--------|
| policy vs random | 20 | 16 / 4 / 0 | 90.0% | +382 (+255 to +761) | score >= 98% |
| policy-safe vs random | 20 | 20 / 0 / 0 | 100.0% | +inf (+287 to +inf) | score >= 98% |
| MCTS 400 vs policy | 20 | 20 / 0 / 0 | 100.0% | +inf (+287 to +inf) | score > 55% |

Puzzles, first move: 500 from lichess_puzzles_500.csv.

| Player | <=1000 | 1001-1500 | 1501-2000 | >2000 | All |
|--------|---:|---:|---:|---:|---:|
| policy | 84.8% | 73.6% | 62.4% | 44.0% | 66.2% |
| mcts-200 | 99.2% | 91.2% | 82.4% | 69.6% | 85.6% |
<!-- end eval: base-best quick -->

<!-- eval: base-best full -->
### base-best.pt, full suite

2026-11-27, 2 x Tesla T4 (cuda, 4 worker(s)), chessnet 0.2.0.dev0. 50 balanced openings, each played with both colours.

| Match | Games | W / D / L | Score | Elo (95% CI) | Target |
|-------|------:|-----------|------:|--------------|--------|
| policy vs random | 100 | 95 / 5 / 0 | 97.5% | +636 (+525 to +975) | score >= 98% |
| policy-safe vs random | 100 | 100 / 0 / 0 | 100.0% | +inf (+566 to +inf) | score >= 98% |
| MCTS 400 vs policy | 100 | 100 / 0 / 0 | 100.0% | +inf (+566 to +inf) | score > 55% |
| MCTS 800 vs Stockfish UCI_Elo 1320 | 50 | 50 / 0 / 0 | 100.0% | +inf (+446 to +inf) | non-zero score |
| MCTS 800 vs Stockfish Skill Level 0 | 50 | 50 / 0 / 0 | 100.0% | +inf (+446 to +inf) | - |
| wide-best vs base, 800 playouts each | 50 | 12 / 25 / 13 | 49.0% | -7 (-76 to +62) | - |
| wide-best vs base, 1 s per move each | 50 | 13 / 29 / 8 | 55.0% | +35 (-27 to +99) | - |
| wide-best vs base, 1 s per move each (laptop CPU) | 20 | 0 / 2 / 18 | 5.0% | -512 (-inf to -353) | - |

Puzzles, first move: 2000 sampled from lichess_db_puzzle.csv.zst (seed 0).

| Player | <=1000 | 1001-1500 | 1501-2000 | >2000 | All |
|--------|---:|---:|---:|---:|---:|
| policy | 82.8% | 69.2% | 55.6% | 49.2% | 64.2% |
| mcts-800 | 99.4% | 98.0% | 92.2% | 84.6% | 93.5% |

The Kaggle run hung after MCTS 800 vs Stockfish UCI_Elo 1320 (fixed since). That match's results were recovered from the kernel log; its PGN was lost.

policy-safe vs random was played on 11th Gen Intel(R) Core(TM) i3-1115G4 @ 3.00GHz (cpu), 1 game at a time.

wide-best vs base, 1 s per move each was played 2 games at a time.

wide-best vs base, 1 s per move each (laptop CPU) was played on 11th Gen Intel(R) Core(TM) i3-1115G4 @ 3.00GHz (cpu), 1 game at a time.
<!-- end eval: base-best full -->

### What the full suite shows

- **Against a random mover** the raw policy scores 97.5%: 95 wins and 5 draws (4
  stalemates, 1 threefold repetition), half a point short of the 98% target. Policy-safe,
  the engine's policy mode, wins all 100 games.
- **Search over the policy:** MCTS with 400 playouts wins all 100 games against the raw
  policy (target: above 55%).
- **Stockfish at its weakest:** in Stockfish 16, `UCI_Elo 1320` converts to skill level
  0.0, the same setting as `Skill Level 0`, so the two matches face the same opponent. Their
  games differ because Stockfish's weakened move choice is random (seeded from the clock):
  no game of one match has the result and length of the same game in the other. MCTS 800
  won all 100 games, each by checkmate, which puts it at least 566 Elo above that level
  (95%). That is a floor, as the target (a non-zero score) intended, not a rating.
- **Wide vs base at equal playouts** (800 each, GPU): 12 / 25 / 13, no measurable
  difference (-7 Elo, -76 to +62). Per playout, wide's 2 epochs give as strong a network
  as base's 5.
- **Wide vs base at equal time:** on a T4, one game per GPU at 1 s per move each, wide
  scores 55% (+35, -27 to +99, not significant), most likely because the Python search
  loop, not the network, sets the pace on the GPU. On the laptop CPU wide loses 18 of 20
  (5%, -512, -inf to -353): there the network is most of the cost, and wide completes
  about 257 playouts in a 1 s move against base's 490. On a CPU, base is the network to
  play with.
- **Puzzles:** search lifts first-move accuracy from 64.2% (policy) to 93.5% (MCTS 800),
  and the gain grows with difficulty: +16.6 points at 1000 and below, +28.8 at 1001-1500,
  +36.6 at 1501-2000 and +35.4 above 2000. The policy alone falls from 82.8% to 49.2%
  across the buckets; MCTS 800 still finds the first move of 84.6% of puzzles rated above
  2000. Each bucket has 500 puzzles, so each figure is within about +-0.7 to +-4.4 points
  (95%).
