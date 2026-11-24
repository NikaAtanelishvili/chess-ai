"""Hand-picked positions (FEN) used across the encoding tests."""

import chess

START = chess.STARTING_FEN

# Both sides may castle on both wings immediately.
CASTLING_WHITE = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
CASTLING_BLACK = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1"

# White to move may capture en passant (exf6) after 1.e4 d5 2.e5 f5.
EN_PASSANT_WHITE = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
# Black to move may capture en passant (exf3) after White's f2-f4.
EN_PASSANT_BLACK = "rnbqkbnr/pppp1ppp/8/8/3PpP2/8/PPP1P1PP/RNBQKBNR b KQkq f3 0 3"

# Pawn on the 7th (2nd) rank that can promote straight or by capturing on either side.
PROMOTION_WHITE = "r1n1k3/1P6/8/8/8/8/8/4K3 w - - 0 1"
PROMOTION_BLACK = "4k3/8/8/8/8/8/1p6/R1N1K3 b - - 0 1"

# Known maximum-mobility position: 218 legal moves for White.
MAX_MOBILITY = "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1"
MAX_MOBILITY_MOVES = 218

# Starting points for random playouts that reach promotions and castling often.
PROMOTION_RICH = "1n1q1b1r/PPPPPPPP/8/k7/7K/8/pppppppp/1N1Q1B1R w - - 0 1"
CASTLING_RICH = "r3k2r/p1pp1pp1/8/8/8/8/P1PP1PP1/R3K2R w KQkq - 0 1"

ALL_FIXTURES = (
    START,
    CASTLING_WHITE,
    CASTLING_BLACK,
    EN_PASSANT_WHITE,
    EN_PASSANT_BLACK,
    PROMOTION_WHITE,
    PROMOTION_BLACK,
    MAX_MOBILITY,
    PROMOTION_RICH,
    CASTLING_RICH,
)

# Start positions for generated games (property tests and the coverage sample).
PLAYOUT_STARTS = (
    START,
    PROMOTION_RICH,
    CASTLING_RICH,
    EN_PASSANT_WHITE,
    EN_PASSANT_BLACK,
)
