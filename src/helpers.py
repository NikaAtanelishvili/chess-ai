import math
import random

import numpy as np

class BitboardManipulations:
    @staticmethod
    def set_bit(bb: np.uint64, square: int) -> np.uint64:
        """Set the bit at 'square' in bitboard 'bb'."""
        return bb | np.uint64(1) << np.uint64(square)

    @staticmethod
    def clear_bit(bb: np.uint64, square: int) -> np.uint64:
        """Clear the bit at 'square' in bitboard 'bb'."""
        return bb & ~(np.uint64(1) << np.uint64(square))

    @staticmethod
    def is_bit_set(bb: np.uint64, square: int) -> bool:
        """Check if a bit at 'square' is set in 'bb'."""
        return (bb & (np.uint64(1) << np.uint64(square))) != np.uint64(0)


    @staticmethod
    def bit_scan(bitboard: int):
        """
        Generator that yields the indices of set bits (1s) in the bitboard.
        (Least-significant bit first.)

        UNOPTIMIZED!
        while bitboard: # Loops while there are still set bits (1s) in the bitboard
            lsb = bitboard & -bitboard # Isolates the rightmost 1 while setting all other bits to 0
            square = lsb.bit_length() - 1
            # bit_length() returns the index of the highest bit set (1-based).
            # Subtracting 1 makes it 0-based.
            yield square
            bitboard &= bitboard - 1 # Clearing the Processed Bit
            # bitboard - 1 flips all bits after the rightmost 1 (including the 1 itself).
            # bitboard &= bitboard - 1 effectively removes the lowest set bit.
            ###EXAMPLE##
            # Iteration ==> 1
            # bitboard (Binary) ==> 10010100 (148)
            # lsb = bitboard & -bitboard ==> 00000100 (4)
            # square = lsb.bit_length() - 1 ==> 2
            # New bitboard ==> 10010000 (144)
        """
        # math.log2(x) is implemented efficiently in hardware (single CPU instruction in many cases).
        # This avoids the need for bit_length() and subtraction.
        while bitboard:
            square = int(np.log2(bitboard & -bitboard))  # Compute least significant bit index
            yield square
            bitboard &= bitboard - np.uint64(1)  # Clear the least significant bit


def generate_occupancy_subsets(mask):
    """
    Generates all possible occupancies (subsets) of a given bitboard mask
    Each subset represents a different combination of occupied squares within the mask.

    mask: A bitboard (64-bit integer) representing a set of squares
    (e.g., squares that could contain blocking pieces for a rook)
    """
    sub = 0
    while True:
        yield sub
        sub = (sub - mask) & mask
        # Subtract mask from sub rolls over the bits, kind of like counting in binary
        # ~Binary subtraction works similarly to decimal subtraction,
        # but it's often implemented using two’s complement arithmetic in computers.~
        #
        # ==EXAMPLE== (3-5=?)
        # Convert -5 to Two’s Complement
        # Write 5 in binary:
        # 5 = 0b0101
        # Invert the bits (one’s complement):
        # 1010
        # Add 1 to get -5:
        # 1011   (This is -5 in two’s complement)
        # Step 2: Add Instead of Subtract
        # Now, instead of doing 3 - 5, we do:
        #    0011   (3)
        # +  1011   (-5 in two’s complement)
        # ------------
        #    1110   (-2 in two’s complement) ✅
        # This result (0b1110) represents -2 in two’s complement.
        #
        # Key takeaway: When subtraction "underflow's", we wrap around to a negative value using two’s complement.
        # &(AND) ensures that only bits from mask are kept (all others are reset to 0).
        if sub == 0:
            break
        # Think of this process like counting in binary


import os

# needs update for non-sliding pieces
def save_data(magics_fn, attacks_fn, shifts_fn, masks_fn, magics, attacks, shifts, masks):
    # Define the directories for each type of data
    magics_dir = 'data/magics'
    attacks_dir = 'data/attacks'
    shifts_dir = 'data/shifts'
    masks_dir = 'data/masks'

    # Create directories if they don't exist
    os.makedirs(magics_dir, exist_ok=True)
    os.makedirs(attacks_dir, exist_ok=True)
    os.makedirs(shifts_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Save magic numbers
    np.save(os.path.join(magics_dir, magics_fn), np.array(magics, dtype=np.uint64))

    # Save shifts
    np.save(os.path.join(shifts_dir, shifts_fn), np.array(shifts, dtype=np.uint64))

    # Save masks
    np.save(os.path.join(masks_dir, masks_fn), np.array(masks, dtype=np.uint64))

    # Convert attack tables into a dictionary
    attack_dict = {f"square_{sq}": np.array(attacks[sq], dtype=np.uint64) for sq in range(64)}

    # Save attack tables using .npz (compressed format)
    np.savez_compressed(os.path.join(attacks_dir, attacks_fn), **attack_dict)

    print("Data saved successfully in separate directories.")

#
# def generate_attack_table(square, magic, relevant_bits, mask, generate_attacks):
#     """
#     Generic function to build an attack table for either rooks or bishops.
#     """
#     size = 1 << relevant_bits
#     attacks_array = [0] * size
#
#     # Precompute occupancy -> index -> attacks
#     for occ in generate_occupancy_subsets(mask):
#         index = ((occ * magic) & 0xFFFFFFFFFFFFFFFF) >> (64 - relevant_bits)
#         # Compute the actual attacks given 'occ'
#         attacks_array[index] = generate_attacks(square, occ)
#
#     return attacks_array
