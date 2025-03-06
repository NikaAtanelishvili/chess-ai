import os
import random

import chess
import numpy as np

from src.helpers import set_bit, is_bit_set, generate_occupancy_subsets, save_data, bit_scan

BISHOP_MASKS = [0] * 64
BISHOP_MAGICS = [0] * 64
BISHOP_SHIFTS = [0] * 64
BISHOP_ATTACKS = [0] * 64

def generate_bishop_attacks(square, blockers):
    """
    Compute bishop attacks for 'square' given a bitboard 'blockers'
    that marks occupied squares on the board.

    function returns a bitboard where set bits (1s) represent squares the bishop can attack.
    """
    attacks = 0
    rank, file = divmod(square, 8)

    # Northeast (↗️)
    r, f = rank + 1, file + 1
    while r < 8 and f < 8:
        sq = r * 8 + f
        attacks = set_bit(attacks, sq)
        if is_bit_set(blockers, sq):
            break
        r += 1
        f += 1

    # Northwest (↖️)
    r, f = rank + 1, file - 1
    while r < 8 and f >= 0:
        sq = r * 8 + f
        attacks = set_bit(attacks, sq)
        if is_bit_set(blockers, sq):
            break
        r += 1
        f -= 1

    # Southeast (↘️)
    r, f = rank - 1, file + 1
    while r >= 0 and f < 8:
        sq = r * 8 + f
        attacks = set_bit(attacks, sq)
        if is_bit_set(blockers, sq):
            break
        r -= 1
        f += 1

    # Southwest (↙️)
    r, f = rank - 1, file - 1
    while r >= 0 and f >= 0:
        sq = r * 8 + f
        attacks = set_bit(attacks, sq)
        if is_bit_set(blockers, sq):
            break
        r -= 1
        f -= 1

    return attacks

def generate_bishop_occupancy_mask(square):
    """
    For a bishop on 'square'(0-64), compute the 'relevant occupancy mask'
    (the squares that can block the bishop).

    -The edge squares don’t contribute to attack variations because
    the bishop would always move up to the board’s edge.
    """
    mask = 0
    rank, file = divmod(square, 8)

    # Northeast (↗️)
    r, f = rank + 1, file + 1
    while r < 7 and f < 7:
        sq = r * 8 + f
        mask = set_bit(mask, sq)
        r += 1
        f += 1

    # Northwest (↖️)
    r, f = rank + 1, file - 1
    while r < 7 and f > 0:
        sq = r * 8 + f
        mask = set_bit(mask, sq)
        r += 1
        f -= 1

    # Southeast (↘️)
    r, f = rank - 1, file + 1
    while r > 0 and f < 7:
        sq = r * 8 + f
        mask = set_bit(mask, sq)
        r -= 1
        f += 1

    # Southwest (↙️)
    r, f = rank - 1, file - 1
    while r > 0 and f > 0:
        sq = r * 8 + f
        mask = set_bit(mask, sq)
        r -= 1
        f -= 1


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
        bishop_att = generate_bishop_attacks(square, occ)
        occupancies.append(occ)
        attacks_table.append(bishop_att)

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
            print(magic)
            return magic

# ---------------------------------------------------------------------
# 5) Generating the Attack table (Move table) for the bishop
# ---------------------------------------------------------------------
def generate_bishop_attack_table(square, magic, relevant_bits, mask):
    """
    Build the actual 'ROOK_ATTACKS[square]' table using the found magic.


    """
    size = 1 << relevant_bits
    attacks_array = [0] * size

    # Precompute occupancy -> index -> attacks
    for occ in generate_occupancy_subsets(mask):
        index = ((occ * magic) & 0xFFFFFFFFFFFFFFFF) >> (64 - relevant_bits)
        # Compute the actual bishop attacks given 'occ'
        attacks_array[index] = generate_bishop_attacks(square, occ)

    return attacks_array

# --------------------------------------------------------------------------------------------
# 6) Initializing the global variables: BISHOP_MAGICS, BISHOP_ATTACK, BISHOP_SHIFTS, BISHOP_MASKS
# --------------------------------------------------------------------------------------------
def initialize_bishop_data():
    """
    initialize the bishop's data: magic numbers, attacks table, shifts and masks.
    """
    print('cuh')
    for square in range(64):
        # 1) Compute the relevant mask for this square
        mask = generate_bishop_occupancy_mask(square)
        BISHOP_MASKS[square] = mask

        # 2) Count how many bits are set -> relevant_bits
        relevant_bits = bin(mask).count('1')
        BISHOP_SHIFTS[square] = 64 - relevant_bits

        # 3) Find a magic number that yields no collisions
        magic = find_magic_number(square, relevant_bits, mask)
        BISHOP_MAGICS[square] = magic

        # 4) Build the final attack table
        BISHOP_ATTACKS[square] = generate_bishop_attack_table(square, magic, relevant_bits, mask)

    save_data(
        'bishop_magic_numbers.npy',
        'bishop_attack_table.npz',
        'bishop_shifts.npy',
        'bishop_masks.npy',
        BISHOP_MAGICS, BISHOP_ATTACKS, BISHOP_SHIFTS, BISHOP_MASKS)

# -------------------------------------------------------------------------
# 7) Generating the bishop's moves (bitboard to chess's Algebraic notation)
# -------------------------------------------------------------------------
def generate_bishop_moves(board: chess.Board, color: bool):
    moves = []

    # 1) All bishops for this color
    bishops = int(board.pieces(chess.ROOK, color))

    # 2) Occupancy bitboards
    occupancy = int(board.occupied)              # all pieces
    own_occ = int(board.occupied_co[color])      # own pieces only

    for square in bit_scan(bishops):
        # 3) Compute relevant occupancy
        relevant_occ = occupancy & BISHOP_MASKS[square]

        # 4) Multiply by magic number and shift
        magic = BISHOP_MAGICS[square]
        shift = BISHOP_SHIFTS[square]
        index = (relevant_occ * magic) >> shift

        # 5) Look up the attacked squares bitboard
        attacks = BISHOP_ATTACKS[square][index]

        # 6) Remove squares occupied by our own pieces
        valid_attacks = attacks & ~own_occ

        # 7) Convert bitboard to moves
        for target in bit_scan(valid_attacks):
            moves.append(chess.Move(square, target))

    return moves

def load_data():
    """Loads magic numbers and attack tables from the appropriate directories."""
    global BISHOP_MAGICS, BISHOP_ATTACKS, BISHOP_SHIFTS, BISHOP_MASKS

    # Define the directories for each type of data
    magics_dir = 'data/magics'
    attacks_dir = 'data/attacks'
    shifts_dir = 'data/shifts'
    masks_dir = 'data/masks'

    # Load magic numbers
    BISHOP_MAGICS = np.load(os.path.join(magics_dir, "bishop_magic_numbers.npy"))

    # Load shifts
    BISHOP_ATTACKS = np.load(os.path.join(shifts_dir, "bishop_shifts.npy"))

    # Load masks
    BISHOP_SHIFTS = np.load(os.path.join(masks_dir, "bishop_masks.npy"))

    # Load attack tables from .npz
    attack_data = np.load(os.path.join(attacks_dir, "bishop_attack_table.npz"))
    BISHOP_MASKS = {int(sq.split("_")[1]): attack_data[sq] for sq in attack_data.files}

    print("Magic data loaded successfully.")

paths = {
    'magics': 'data/magics/bishop_magic_numbers.npy',
    'shifts': 'data/shifts/bishop_shifts.npy',
    'masks': 'data/masks/bishop_masks.npy',
    'attacks': 'data/attacks/bishop_attack_table.npz',
}

# Check if any required file is missing
missing_files = [name for name, path in paths.items() if not os.path.exists(path)]

if missing_files:
    # If there are missing files, initialize bishop data and raise an error
    print('initializing the data')
    initialize_bishop_data()
else:
    # If all files are found, load the data
    load_data()
    print(BISHOP_MASKS)





if '__main__' == __name__:
    initialize_bishop_data()
    board = chess.Board()
    generate_bishop_moves(board, True)

