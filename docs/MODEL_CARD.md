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
