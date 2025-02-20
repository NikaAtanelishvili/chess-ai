import random

import chess
import numpy as np
import os

from src.helpers import set_bit, is_bit_set, bit_scan, generate_occupancy_subsets, save_data

# ---------------------------------------------------------------------
# 1) Global Tables
# ---------------------------------------------------------------------
ROOK_MASKS = [0] * 64          # Relevant occupancy mask for each square
ROOK_MAGICS = [0] * 64         # Magic number for each square
ROOK_SHIFTS = [0] * 64         # Shift = 64 - number_of_relevant_bits
ROOK_ATTACKS = [0] * 64    # Each entry is a list of bitboards

# ---------------------------------------------------------------------
# 2) Helper Functions
# ---------------------------------------------------------------------
def generate_rook_attacks(square, blockers):
    """
    Compute rook attacks for 'square' given a bitboard 'blockers'
    that marks occupied squares on the board.

    function returns a bitboard where set bits (1s) represent squares the rook can attack.
    """
    attacks = 0
    rank, file = divmod(square, 8)

    # Up
    for r in range(rank+1, 8):
        sq = r * 8 + file
        attacks = set_bit(attacks, sq)
        if is_bit_set(blockers, sq):
            break

    # Down
    for r in range(rank-1, -1, -1):
        sq = r * 8 + file
        attacks = set_bit(attacks, sq)
        if is_bit_set(blockers, sq):
            break

    # Right
    for f in range(file+1, 8):
        sq = rank * 8 + f
        attacks = set_bit(attacks, sq)
        if is_bit_set(blockers, sq):
            break

    # Left
    for f in range(file-1, -1, -1):
        sq = rank * 8 + f
        attacks = set_bit(attacks, sq)
        if is_bit_set(blockers, sq):
            break

    return attacks

# ---------------------------------------------------------------------
# 3) Generating the Relevant Occupancy Mask
# ---------------------------------------------------------------------
def generate_rook_occupancy_mask(square):
    """
    For a rook on 'square'(0-64), compute the 'relevant occupancy mask'
    (the squares that can block the rook).

    -The edge squares don’t contribute to attack variations because
    the rook would always move up to the board’s edge.
    """
    mask = 0
    rank, file = divmod(square, 8)

    # Up (exclude final rank edge)
    for r in range(rank+1, 7):
        sq = r * 8 + file
        mask = set_bit(mask, sq)

    # Down (exclude final rank edge)
    for r in range(rank-1, 0, -1):
        sq = r * 8 + file
        mask = set_bit(mask, sq)

    # Right (exclude final file edge)
    for f in range(file+1, 7):
        sq = rank * 8 + f
        mask = set_bit(mask, sq)

    # Left (exclude final file edge)
    for f in range(file-1, 0, -1):
        sq = rank * 8 + f
        mask = set_bit(mask, sq)

    return mask

# ---------------------------------------------------------------------
# 4) Finding a Magic Number
# ---------------------------------------------------------------------
def find_magic_number(square, relevant_bits, mask):
    """
    Brute-force search for a magic number for 'square' that maps
    every possible occupancy to a unique index.
    """
    # We'll need up to 2^relevant_bits entries
    size = 1 << relevant_bits

    occupancies = []
    attacks_table = []

    # 1) Build all occupancies for this mask and store their attacks
    for occ in generate_occupancy_subsets(mask):
        rook_att = generate_rook_attacks(square, occ)
        occupancies.append(occ)
        attacks_table.append(rook_att)

    # 2) Try random 64-bit numbers until we find a collision-free magic
    while True:
        magic = random.getrandbits(64) & random.getrandbits(64) & random.getrandbits(64)

        # "good" magic number needs high bit entropy (spread-out bits for better indexing)
        if magic & 0xFF00000000000000 == 0: #isolates the top 8 bits and checks if the top 8 bits of magic are all zeros
            continue

        used = [-1] * size
        fail = False

        for i, occ in enumerate(occupancies):
            index = ((occ * magic) & 0xFFFFFFFFFFFFFFFF) >> (64 - relevant_bits)
            # occ * magic:
            # -This multiplies the occupancy bitboard (occ) by the magic number (magic).
            # -The multiplication causes a pseudo-random spread of bits.
            #
            # & 0xFFFFFFFFFFFFFFFF
            # -This ensures the result is confined to 64 bits (in case of overflow).
            # -Equivalent to taking the lower 64 bits of the multiplication.
            #
            # >> (64 - relevant_bits)
            # -Right-shifts the result to extract the most significant bits.
            # -relevant_bits is typically ≤12, meaning we extract the topmost 12 bits.
            # Why does this work?
            # Multiplying by magic spreads bits unpredictably, and extracting the top relevant_bits ensures that
            # different occupancies map to unique indices.
            if used[index] == -1:
                used[index] = attacks_table[i]
            elif used[index] != attacks_table[i]:
                fail = True
                break

        if not fail:
            return magic

# ---------------------------------------------------------------------
# 5) Generating the Attack table (Move table) for the rook
# ---------------------------------------------------------------------
def generate_rook_attack_table(square, magic, relevant_bits, mask):
    """
    Build the actual 'ROOK_ATTACKS[square]' table using the found magic.
    """
    size = 1 << relevant_bits
    attacks_array = [0] * size

    # Precompute occupancy -> index -> attacks
    for occ in generate_occupancy_subsets(mask):
        index = ((occ * magic) & 0xFFFFFFFFFFFFFFFF) >> (64 - relevant_bits)
        # Compute the actual rook attacks given 'occ'
        attacks_array[index] = generate_rook_attacks(square, occ)

    return attacks_array

# --------------------------------------------------------------------------------------------
# 6) Initializing the global variables: ROOKS_MAGICS, ROOKS_ATTACK, ROOKS_SHIFTS, ROOKS_MASKS
# --------------------------------------------------------------------------------------------
def initialize_rook_data():
    """
    initialize the rook's data: magic numbers, attacks table, shifts and masks.
    """
    for square in range(64):
        # 1) Compute the relevant mask for this square
        mask = generate_rook_occupancy_mask(square)
        ROOK_MASKS[square] = mask

        # 2) Count how many bits are set -> relevant_bits
        relevant_bits = bin(mask).count('1')
        ROOK_SHIFTS[square] = 64 - relevant_bits

        # 3) Find a magic number that yields no collisions
        magic = find_magic_number(square, relevant_bits, mask)
        ROOK_MAGICS[square] = magic

        # 4) Build the final attack table
        ROOK_ATTACKS[square] = generate_rook_attack_table(square, magic, relevant_bits, mask)

    save_data(
        'rook_magic_numbers.npy',
        'rook_attack_table.npz',
        'rook_shifts.npy',
        'rook_masks.npy',
        ROOK_MAGICS, ROOK_ATTACKS, ROOK_SHIFTS, ROOK_MASKS)

# -------------------------------------------------------------------------
# 7) Generating the rook's moves (bitboard to chess's Algebraic notation)
# -------------------------------------------------------------------------
def generate_rook_moves(board: chess.Board, color: bool):
    moves = []

    # 1) All rooks for this color
    rooks = int(board.pieces(chess.ROOK, color))

    # 2) Occupancy bitboards
    occupancy = int(board.occupied)              # all pieces
    own_occ = int(board.occupied_co[color])      # own pieces only

    for square in bit_scan(rooks):
        # 3) Compute relevant occupancy
        relevant_occ = occupancy & ROOK_MASKS[square]

        # 4) Multiply by magic number and shift
        magic = ROOK_MAGICS[square]
        shift = ROOK_SHIFTS[square]
        index = (relevant_occ * magic) >> shift

        # 5) Look up the attacked squares bitboard
        attacks = ROOK_ATTACKS[square][index]

        # 6) Remove squares occupied by our own pieces
        valid_attacks = attacks & ~own_occ

        # 7) Convert bitboard to moves
        for target in bit_scan(valid_attacks):
            moves.append(chess.Move(square, target))

    return moves

def load_data():
    """Loads magic numbers and attack tables from the appropriate directories."""
    global ROOK_MAGICS, ROOK_ATTACKS, ROOK_SHIFTS, ROOK_MASKS

    # Define the directories for each type of data
    magics_dir = 'data/magics'
    attacks_dir = 'data/attacks'
    shifts_dir = 'data/shifts'
    masks_dir = 'data/masks'

    # Load magic numbers
    ROOK_MAGICS = np.load(os.path.join(magics_dir, "rook_magic_numbers.npy"))

    # Load shifts
    ROOK_SHIFTS = np.load(os.path.join(shifts_dir, "rook_shifts.npy"))

    # Load masks
    ROOK_MASKS = np.load(os.path.join(masks_dir, "rook_masks.npy"))

    # Load attack tables from .npz
    attack_data = np.load(os.path.join(attacks_dir, "rook_attack_table.npz"))
    ROOK_ATTACKS = {int(sq.split("_")[1]): attack_data[sq] for sq in attack_data.files}

    print("Magic data loaded successfully.")

paths = {
    'magics': 'data/magics/rook_magic_numbers.npy',
    'shifts': 'data/magics/rook_shifts.npy',
    'masks': 'data/magics/rook_masks.npy',
    'attacks': 'data/magics/rook_attack_table.npz',
}

# Check if any required file is missing
missing_files = [name for name, path in paths.items() if not os.path.exists(path)]

if missing_files:
    # If there are missing files, initialize rook data and raise an error
    initialize_rook_data()
    raise FileNotFoundError(f"Missing {', '.join(missing_files)} file(s) are now created!")
else:
    # If all files are found, load the data
    load_data()

# ---------------------------------------------------------------------
# 8) Usage Example
# ---------------------------------------------------------------------

# if __name__ == "__main__":
#     generate_and_save_magic_data()
