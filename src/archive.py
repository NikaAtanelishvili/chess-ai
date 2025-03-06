import math
import chess

KNIGHT_MOVES = {}
KING_MOVES = {}

# True for White, False for Black
PAWN_SINGLE_MOVES = {True: [0] * 64, False: [0] * 64}
PAWN_DOUBLE_MOVES = {True: [0] * 64, False: [0] * 64}
PAWN_ATTACKS = {True: [0] * 64, False: [0] * 64}


# KNIGHT | GENERATES ILLEGAL AND LEGAL MOVES
def init_knight_moves():
    global KNIGHT_MOVES

    for square in chess.SQUARES:
        moves = 0
        rank, file = divmod(square, 8)  # instead of nested loop

        # Knight move offsets as (delta_rank, delta_file)
        offsets = [(2, 1), (2, -1), (-2, 1), (-2, -1),
                   (1, 2), (1, -2), (-1, 2), (-1, -2)]

        for dr, df in offsets:
            nr, nf = rank + dr, file + df # new rank and file
            if 0 <= nr < 8 and 0 <= nf < 8:
                moves |= 1 << (nr * 8 + nf) # bitboard representation

            KNIGHT_MOVES[square] = moves

# KING | GENERATES ILLEGAL AND LEGAL MOVES
def init_king_moves():
    global KING_MOVES

    for square in chess.SQUARES:
        moves = 0
        rank, file = divmod(square, 8)

        offsets = [(1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dr, df in offsets:
            nr, nf = rank + dr, file + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                moves |= 1 << (nr * 8 + nf)
                # (nr * 8 + nf) computes the target square index (0–63).
                # 1 << (nr * 8 + nf) shifts the binary 1 to the target square’s position.
                # moves |= ... sets the bit at that position in moves.
        KING_MOVES[square] = moves

# PAWN | GENERATES ILLEGAL AND LEGAL MOVES
def init_pawn_moves():
    """
    Precompute pawn move tables for white and black.

    On an empty board:
      - A white pawn can push one square forward (if not on rank 8).
      - From its initial rank (rank 2, index 1) a white pawn can push two squares.
      - Its capture moves are one square diagonally (taking care of file boundaries).

    Black moves are the mirror image.
    """
    for square in range(64):
        rank, file = divmod(square, 8)

        # WHITE moves:
        white_single = 0
        white_double = 0
        white_attacks = 0

        if rank < 7:  # Can always push one square if not on last rank.
            white_single |= 1 << (square + 8) # Single-square push
            # Diagonal captures (avoid wrap‐around on files).
            if file > 0:
                white_attacks |= 1 << (square + 7) # Left capture
            if file < 7:
                white_attacks |= 1 << (square + 9) # Right capture
            # From the initial rank (rank index 1 = 2nd rank), two‐square push is possible.
            if rank == 1:
                white_double |= 1 << (square + 16) # double-square push

        PAWN_SINGLE_MOVES[True][square] = white_single
        PAWN_DOUBLE_MOVES[True][square] = white_double
        PAWN_ATTACKS[True][square] = white_attacks

        # BLACK moves:
        black_single = 0
        black_double = 0
        black_attacks = 0

        if rank > 0:  # Can push one square backward (downwards)
            black_single |= 1 << (square - 8)
            # Diagonal captures for black:
            if file > 0:
                black_attacks |= 1 << (square - 9)
            if file < 7:
                black_attacks |= 1 << (square - 7)
            # From the initial rank for black (rank index 6 = 7th rank)
            if rank == 6:
                black_double |= 1 << (square - 16)

        PAWN_SINGLE_MOVES[False][square] = black_single
        PAWN_DOUBLE_MOVES[False][square] = black_double
        PAWN_ATTACKS[False][square] = black_attacks

# QUEEN, BISHOP, ROOK | GENERATES ONLY LEGAL MOVES
def generate_slider_moves(board: chess.Board, square: int, color: bool, directions):
    moves = []

    occupancy = board.occupied

    own_occ = 0 # bitboard of all squares occupied by our own pieces.
    for pt in chess.PIECE_TYPES:
        own_occ |= int(board.pieces(pt, color)) # bitboard of all pieces of type pt for color

    rank, file = divmod(square, 8)

    for dr, df in directions:
        r, f = rank, file

        while True:
            r += dr
            f += df
            if not (0 <= r < 8 and 0 <= f < 8):
                break
            target = r * 8 + f
            target_bit = 1 << target
            if own_occ & target_bit:
                # Our own piece is blocking the ray.
                break
            moves.append(chess.Move(square, target))
            if occupancy & target_bit:
                # There's an enemy piece; can capture but not go beyond.
                break
    return moves

init_pawn_moves()
init_knight_moves()
init_king_moves()





# -----------------------------------------------------------------------------
# Bitboard Move Generation for Each Piece Type
# -----------------------------------------------------------------------------

def generate_knight_moves_bitboard(board: chess.Board, color: bool):
    moves = []

    own_occ = 0

    for pt in chess.PIECE_TYPES:
        own_occ |= int(board.pieces(pt, color))

    knights = int(board.pieces(chess.KNIGHT, color))

    for square in bit_scan(knights):
        # ~own_occ - bitmask of empty and opponent-occupied squares
        candidate_bb = KNIGHT_MOVES[square] & ~own_occ #  filters out squares occupied by the player's own pieces
        for target in bit_scan(candidate_bb): # valid target squares
            moves.append(chess.Move(square, target))

    return moves


def generate_king_moves_bitboard(board: chess.Board, color: bool):
    moves = []
    own_occ = 0

    for pt in chess.PIECE_TYPES:
        own_occ |= int(board.pieces(pt, color))

    kings = int(board.pieces(chess.KING, color))

    for square in bit_scan(kings):
        candidate_bb = KING_MOVES[square] & ~own_occ
        for target in bit_scan(candidate_bb):
            moves.append(chess.Move(square, target))

    return moves


def generate_rook_moves_bitboard(board: chess.Board, color: bool):
    moves = []
    rooks = int(board.pieces(chess.ROOK, color))
    # Rook directions: horizontal & vertical.
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for square in bit_scan(rooks):
        moves.extend(generate_slider_moves(board, square, color, directions))
    return moves


def generate_bishop_moves_bitboard(board: chess.Board, color: bool):
    moves = []
    bishops = int(board.pieces(chess.BISHOP, color))
    # Bishop directions: diagonals.
    directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for square in bit_scan(bishops):
        moves.extend(generate_slider_moves(board, square, color, directions))
    return moves


def generate_queen_moves_bitboard(board: chess.Board, color: bool):
    moves = []
    queens = int(board.pieces(chess.QUEEN, color))
    # Queen combines rook and bishop directions.
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for square in bit_scan(queens):
        moves.extend(generate_slider_moves(board, square, color, directions))
    return moves


def generate_pawn_moves_bitboard(board: chess.Board, color: bool):
    """
    Generate pawn moves using precomputed move tables and bitboard filtering.

    This function handles:
      - Single pawn pushes (only if the destination is empty).
      - Double pushes from the initial rank (if both the intermediate
        and destination squares are empty).
      - Captures (only if the target square is occupied by an enemy piece).
      - Promotions (if a push or capture ends on the promotion rank).
      - En passant captures (if board.ep_square is set).

    The board is assumed to provide:
      - board.pieces(piece_type, color): returns a bitboard (int) of that piece.
      - board.occupied: bitboard of all pieces.
      - board.ep_square: en passant target square (or None if not available).
    """
    moves = []
    pawn_bb = int(board.pieces(chess.PAWN, color))
    # Compute the bitboard of all empty squares.
    empty = ~board.occupied & 0xFFFFFFFFFFFFFFFF

    # Compute opponent occupancy (for capture moves).
    opp_occ = 0
    for pt in chess.PIECE_TYPES:
        opp_occ |= int(board.pieces(pt, not color))

    for square in bit_scan(pawn_bb):
        # --- Single Push Moves ---
        single_push = PAWN_SINGLE_MOVES[color][square] & empty
        for target in bit_scan(single_push):
            # Check for promotion (white promotes on rank 8, black on rank 1).
            if (color and chess.square_rank(target) == 7) or (not color and chess.square_rank(target) == 0):
                # For promotions, generate one move per possible promotion piece.
                for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    moves.append(chess.Move(square, target, promotion=promo))
            else:
                moves.append(chess.Move(square, target))

        # --- Double Push Moves ---
        # Only possible if pawn is on its starting rank.
        if (color and chess.square_rank(square) == 1) or (not color and chess.square_rank(square) == 6):
            # Determine the intermediate and destination squares.
            inter_square = square + 8 if color else square - 8
            dest_square = square + 16 if color else square - 16
            # Both squares must be empty.
            if (empty & (1 << inter_square)) and (empty & (1 << dest_square)):
                moves.append(chess.Move(square, dest_square))

        # --- Capture Moves ---
        pawn_caps = PAWN_ATTACKS[color][square] & opp_occ
        for target in bit_scan(pawn_caps):
            if (color and chess.square_rank(target) == 7) or (not color and chess.square_rank(target) == 0):
                for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    moves.append(chess.Move(square, target, promotion=promo))
            else:
                moves.append(chess.Move(square, target))

        # --- En Passant ---
        if board.ep_square is not None:
            if PAWN_ATTACKS[color][square] & (1 << board.ep_square):
                moves.append(chess.Move(square, board.ep_square))

    return moves


def generate_castling_moves(board: chess.Board, color: bool):
    moves = []
    opponent = not color

    # king's starting square
    king_square = list(board.pieces(chess.KING, color))[0]

    if color:  # White
        # Kingside castling: e1 -> g1
        if board.has_kingside_castling_rights(chess.WHITE):
            # Squares f1 (square 5) and g1 (square 6) must be empty.
            if not board.occupied & ((1 << chess.F1) | (1 << chess.G1)):
                # e1, f1, and g1 must not be attacked.
                if (not board.is_attacked_by(opponent, chess.E1) and
                    not board.is_attacked_by(opponent, chess.F1) and
                    not board.is_attacked_by(opponent, chess.G1)):
                    moves.append(chess.Move(chess.E1, chess.G1))
        # Queen side castling: e1 -> c1
        if board.has_queenside_castling_rights(chess.WHITE):
            # Squares b1 (square 1), c1 (square 2), and d1 (square 3) must be empty.
            if not board.occupied & ((1 << chess.B1) | (1 << chess.C1) | (1 << chess.D1)):
                # e1, d1, and c1 must not be attacked.
                if (not board.is_attacked_by(opponent, chess.E1) and
                    not board.is_attacked_by(opponent, chess.D1) and
                    not board.is_attacked_by(opponent, chess.C1)):
                    moves.append(chess.Move(chess.E1, chess.C1))
    else:  # Black
        # King side castling: e8 -> g8
        if board.has_kingside_castling_rights(chess.BLACK):
            # Squares f8 (square 61) and g8 (square 62) must be empty.
            if not board.occupied & ((1 << chess.F8) | (1 << chess.G8)):
                # e8, f8, and g8 must not be attacked.
                if (not board.is_attacked_by(opponent, chess.E8) and
                    not board.is_attacked_by(opponent, chess.F8) and
                    not board.is_attacked_by(opponent, chess.G8)):
                    moves.append(chess.Move(chess.E8, chess.G8))
        # Queen side castling: e8 -> c8
        if board.has_queenside_castling_rights(chess.BLACK):
            # Squares b8 (square 57), c8 (square 58), and d8 (square 59) must be empty.
            if not board.occupied & ((1 << chess.B8) | (1 << chess.C8) | (1 << chess.D8)):
                # e8, d8, and c8 must not be attacked.
                if (not board.is_attacked_by(opponent, chess.E8) and
                    not board.is_attacked_by(opponent, chess.D8) and
                    not board.is_attacked_by(opponent, chess.C8)):
                    moves.append(chess.Move(chess.E8, chess.C8))
    return moves


def generate_moves_bitboard(board: chess.Board, color: bool):
    moves = []
    moves.extend(generate_knight_moves_bitboard(board, color))
    moves.extend(generate_king_moves_bitboard(board, color))
    moves.extend(generate_rook_moves_bitboard(board, color))
    moves.extend(generate_bishop_moves_bitboard(board, color))
    moves.extend(generate_queen_moves_bitboard(board, color))
    moves.extend(generate_pawn_moves_bitboard(board, color))
    moves.extend(generate_castling_moves(board, color))
    # Pawn moves (and special moves like castling/en passant) require additional handling.

    legal_moves = []
    for move in moves:
        board.push(move)
        if not board.is_check():
            legal_moves.append(move)
        board.pop()

    return legal_moves

# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    board = chess.Board()
    # Generate moves for White using our bitboard method.
    white_moves = generate_moves_bitboard(board, chess.WHITE)
    for move in white_moves:
        print(move.uci())
