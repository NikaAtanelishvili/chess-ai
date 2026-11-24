# Test fixtures

## `mini.pgn` / `mini.pgn.zst`

20 games for the data-pipeline tests. `mini.pgn.zst` is `mini.pgn` compressed with zstd
(level 19); a test checks that the two match.

**Source.** Real games from the start of the Lichess open database dump for October 2026
(`lichess_db_standard_rated_2026-10.pgn.zst`), released under CC0.

**Edits.**

- Anonymized: player names are replaced by `whiteNN` / `blackNN`, title headers are removed,
  and game URLs are replaced by `https://lichess.org/game/fixtNNxx`. The suffix `xx` is chosen
  so that game 09 falls into the validation split and game 10 into the test split. All other
  games fall into the training split.
- Ratings are raised to 2000+ in games 08, 13, 16 and 17, so that each rejected game fails
  exactly one check. Game 08 keeps its real underpromotion (`...h1=R`).
- Game 16's correspondence game header is replaced by a blitz time control (only the
  termination should fail).
- Games 18-20 reuse the moves of games 01-03 with one header changed each. These cases do
  not occur in the rated standard dump.

**Expected outcome** with the default filters (rated, both players 2000+, base time at
least 180 s, termination Normal or Time forfeit, result decided or drawn, at least 20 plies,
standard chess):

| # | Event | Time control | Elo (W/B) | Termination | Result | Plies | Outcome | Split |
|---|-------|--------------|-----------|-------------|--------|------:|---------|-------|
| 01 | Rated Blitz | 180+2 | 2103/2125 | Normal | 1-0 | 76 | kept | train |
| 02 | Rated Blitz | 180+0 | 2205/2156 | Normal | 0-1 | 88 | kept (Black win) | train |
| 03 | Rated Rapid | 600+0 | 2087/2189 | Normal | 1/2-1/2 | 111 | kept (draw) | train |
| 04 | Rated Blitz | 300+0 | 2137/2138 | Time forfeit | 0-1 | 62 | kept | train |
| 05 | Rated Rapid | 900+0 | 2504/2359 | Normal | 1-0 | 169 | kept | train |
| 06 | Rated Blitz tournament | 300+0 | 2066/2077 | Normal | 0-1 | 100 | kept | train |
| 07 | Rated Blitz (with `%eval`) | 180+0 | 2019/2014 | Normal | 1-0 | 51 | kept | train |
| 08 | Rated Rapid tournament | 600+2 | 2104/2087* | Normal | 0-1 | 150 | kept (underpromotion) | train |
| 09 | Rated Blitz | 180+2 | 2072/2073 | Normal | 1-0 | 95 | kept | **val** |
| 10 | Rated Blitz | 180+2 | 2247/2248 | Normal | 1-0 | 61 | kept | **test** |
| 11 | Rated Rapid | 480+0 | 2046/2129 | Normal | 1-0 | 55 | kept | train |
| 12 | Rated Bullet | 60+0 | 2341/2330 | Normal | 0-1 | 70 | rejected: time control | |
| 13 | Rated Rapid tournament | 600+0 | 2050/2010* | Abandoned | 0-1 | 0 | rejected: termination | |
| 14 | Rated Blitz tournament | 300+0 | 2022/1982 | Normal | 0-1 | 118 | rejected: Elo | |
| 15 | Rated Blitz | 180+0 | 2252/2280 | Normal | 1-0 | 13 | rejected: too short | |
| 16 | Rated Blitz | 300+0* | 2010/2020* | Rules infraction | 0-1 | 16 | rejected: termination | |
| 17 | Rated Correspondence | - | 2101/2077* | Normal | 0-1 | 30 | rejected: time control | |
| 18 | Casual Blitz* | 180+2 | 2103/2125 | Normal | 1-0 | 76 | rejected: not rated | |
| 19 | Rated Blitz | 180+0 | 2205/2156 | Normal | `*`* | 88 | rejected: result | |
| 20 | Rated Rapid, `Variant "Chess960"`* | 600+0 | 2087/2189 | Normal | 1/2-1/2 | 111 | rejected: variant | |

\* edited header (see above).

Totals: 20 games scanned, 12 pass the header filter, 11 kept. Positions: 1018
(train 862 from 9 games, val 95, test 61). Rejections: time control 2, termination 2,
Elo 1, too short 1, not rated 1, result 1, variant 1.
