import chess

# ─── 1) Build the 73‐action list ──────────────────────────────────────────
DELTAS = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
KNIGHT_DELTAS = [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]
ACTIONS = [] # (dx, dy, is_promotion, promotion_piece)

# Sliding moves: 8 directions × distances 1–7
for dx, dy in DELTAS:
    for dist in range(1, 8):
        ACTIONS.append((dx * dist, dy * dist, False, None))

# Knight jumps: 8
for dx, dy in KNIGHT_DELTAS:
    ACTIONS.append((dx, dy, False, None))

# UNDERPROM_DIRS = [(0,1),(1,1),(-1,1)]  # relative to white; board is
# new (CORRECT = forward + diagonals)
UNDERPROM_DIRS = [(1,0),(1,1),(1,-1)]

for promo_piece in (chess.ROOK, chess.BISHOP, chess.KNIGHT):
    for dx, dy in UNDERPROM_DIRS:
        ACTIONS.append((dx, dy, True, promo_piece))

assert len(ACTIONS) == 73, f"Expected 73 actions, got {len(ACTIONS)}"

POLICY_SIZE = 64 * len(ACTIONS)  # 4672


# ─── 2) move_to_index with Queen‐as‐non-promo + mirroring ────────────────
def move_to_index(move: chess.Move, board: chess.Board) -> int:
    # Mirror into white‐to‐move perspective if needed
    if board.turn == chess.BLACK:
        from_sq = chess.square_mirror(move.from_square)
        to_sq   = chess.square_mirror(move.to_square)
        promotion = move.promotion
    else:
        from_sq, to_sq = move.from_square, move.to_square
        promotion      = move.promotion

    fr_r, fr_c = divmod(from_sq, 8)
    to_r, to_c = divmod(to_sq,   8)
    dx, dy     = to_r - fr_r, to_c - fr_c

    # Only underpromos get promo_flag=True
    is_promo = (promotion is not None and promotion != chess.QUEEN)

    for aid, (adx, ady, promo_flag, promo_piece) in enumerate(ACTIONS):
        if (adx, ady, promo_flag) == (dx, dy, is_promo):
            # match the right piece for underpromos
            if (not promo_flag) or (promotion == promo_piece):
                return from_sq * len(ACTIONS) + aid

    raise ValueError(f"Illegal move for ACTIONS: {move.uci()}")


# ─── 3) index_to_move with queen‐promotion detection ─────────────────────
def index_to_move(index: int, turn: chess.Color) -> chess.Move:
    n        = len(ACTIONS)
    from_sq  = index // n
    aid      = index % n
    dx, dy, promo_flag, promo_piece = ACTIONS[aid]

    fr_r, fr_c = divmod(from_sq, 8)
    to_r, to_c = fr_r + dx, fr_c + dy

    # off‐board?
    if not (0 <= to_r < 8 and 0 <= to_c < 8):
        return None

    to_sq = to_r * 8 + to_c

    # start with underpromo if flagged
    promotion = promo_piece if promo_flag else None

    # detect *queen* promotions: a one‐step pawn move onto rank 7 (canonical)
    # (dx==1, abs(dy)<=1, started on rank 6)
    if not promo_flag and fr_r == 6 and dx == 1 and abs(dy) <= 1:
        promotion = chess.QUEEN

    # build the canonical‐white move
    mv = chess.Move(from_sq, to_sq, promotion=promotion)

    # mirror back for Black
    if turn == chess.BLACK:
        mv = chess.Move(
            chess.square_mirror(mv.from_square),
            chess.square_mirror(mv.to_square),
            promotion=mv.promotion
        )
    return mv


import chess

# ─── assume your corrected ACTIONS, move_to_index, index_to_move here ────
# (make sure UNDERPROM_DIRS = [(1,0),(1,1),(1,-1)] when you build ACTIONS)
# ────────────────────────────────────────────────────────────────────────

def run_move_encoding_tests():
    test_cases = [
        # (UCI,        description,               side,     full FEN)
        ("e2e4",      "normal move",              chess.WHITE, None),
        ("e7e8q",     "white queen promo",        chess.WHITE,
                      "4k3/4P3/8/8/8/8/8/4K3 w - - 0 1"),
        ("d7d8n",     "white knight under-promo", chess.WHITE,
                      "4k3/3P4/8/8/8/8/8/4K3 w - - 0 1"),
        ("d2d1q",     "black queen promo",        chess.BLACK,
                      "4K3/8/8/8/8/8/3p4/4k3 b - - 0 1"),
        ("d2c1r",     "black rook under-promo",   chess.BLACK,
                      "4K3/8/8/8/8/8/3p4/4k3 b - - 0 1"),
    ]

    for uci, desc, side, fen in test_cases:
        board = chess.Board(fen) if fen else chess.Board()
        board.turn = side

        mv        = chess.Move.from_uci(uci)
        idx       = move_to_index(mv, board)
        roundtrip = index_to_move(idx, board.turn)

        assert roundtrip == mv, (
            f"[{desc}] failed: got {roundtrip.uci()} "
            f"(expected {mv.uci()})"
        )

    print("✅ All move→index→move tests passed!")

if __name__ == "__main__":
    run_move_encoding_tests()

