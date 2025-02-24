import os
import random
from abc import ABC
from typing import List, Tuple, Generator

import chess
import numpy as np

from helpers import BitboardManipulations

class SlidingPiece(ABC):
    def __init__(self, piece_type: str, directions: List[Tuple[int, int]]):
        self.directions = directions
        self.piece_type = piece_type

        self.MAGICS = [0] * 64
        self.ATTACKS = [0] * 64
        self.SHIFTS = [0] * 64
        self.MASKS = [0] * 64

    @staticmethod
    def _generate_attacks_in_direction(square: int, blockers: int, direction: Tuple[int, int]) -> int:
        attacks = 0
        dr, df = direction
        rank, file = divmod(square, 8)
        r, f = rank + dr, file + df

        while 0 <= r < 8 and 0 <= f < 8:
            sq = r * 8 + f
            attacks = BitboardManipulations.set_bit(attacks, sq)
            if BitboardManipulations.is_bit_set(blockers, sq):
                break
            r += dr
            f += df

        return attacks

    @staticmethod
    def _generate_mask_in_direction(square: int, direction: Tuple[int, int]) -> int:
        mask = 0
        dr, df = direction
        rank, file = divmod(square, 8)
        r, f = rank + dr, file + df

        while 0 < r < 7 and 0 < f < 7:
            sq = r * 8 + f
            mask = BitboardManipulations.set_bit(mask, sq)
            # if is_bit_set(blockers, sq):
            #     break
            r += dr
            f += df

        return mask

    @staticmethod
    def generate_occupancy_subsets(mask: int) -> Generator[int]:
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

    def generate_attacks(self, square: int, blockers: int) -> int:
        attacks = 0
        for direction in self.directions:
            attacks |= self._generate_attacks_in_direction(square, blockers, direction)
        return attacks

    def generate_occupancy_mask(self, square: int) -> int:
        """
         For a sliding piece on 'square'(0-64), compute the 'relevant occupancy mask'
         (the squares that can block the rook).

         -The edge squares don’t contribute to attack variations because
         the rook would always move up to the board’s edge.
         """
        mask = 0
        for direction in self.directions:
            mask |= self._generate_mask_in_direction(square, direction)
        return mask

    def find_magic_number(self, square: int, relevant_bits: int, mask: int) -> int:
        """
        Find a magic number for the piece on 'square' that maps occupancies to attack sets.

        Args:
            square (int): The square index (0-63).
            relevant_bits (int): Number of bits in the occupancy mask.
            mask (int): Relevant occupancy mask bitboard.

        Returns:
            int: A 64-bit magic number with no collisions.
        """
        size = 1 << relevant_bits

        occupancies = []
        attacks_table = []

        # Generate all possible occupancy subsets and their attacks
        for occ in self.generate_occupancy_subsets(mask):
            att = self.generate_attacks(square, occ)
            occupancies.append(occ)
            attacks_table.append(att)

        # Try random 64-bit numbers until a collision-free magic is found
        while True:
            magic = random.getrandbits(64) & random.getrandbits(64) & random.getrandbits(64)

            # "good" magic number needs high bit entropy (spread-out bits for better indexing)
            if magic & 0xFF00000000000000 == 0:  # isolates the top 8 bits and checks if the top 8 bits of magic are all zeros
                continue

            used = [-1] * size
            fail = False

            # Test the magic number for collisions
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

    def generate_attack_table(self, square: int, magic: int, relevant_bits: int, mask: int) -> List[int]:
        """
        Args:
            square (int): The square index (0-63).
            magic (int): The magic number for this square.
            relevant_bits (int): Number of bits in the occupancy mask.
            mask (int): Relevant occupancy mask bitboard.
        """
        size = 1 << relevant_bits
        attacks_array = [0] * size

        # Precompute occupancy -> index -> attacks
        for occ in self.generate_occupancy_subsets(mask):
            index = ((occ * magic) & 0xFFFFFFFFFFFFFFFF) >> (64 - relevant_bits)
            # Compute the actual rook attacks given 'occ'
            attacks_array[index] = self.generate_attacks(square, occ)

        return attacks_array

    def initialize_data(self) -> None:
        """
        Initialize the piece's data: magic numbers, attacks table, shifts, and masks.
        """
        for square in range(64):
            # 1) Compute the relevant mask for this square
            mask = self.generate_occupancy_mask(square)
            self.MASKS[square] = mask

            # 2) Count how many bits are set -> relevant_bits
            relevant_bits = bin(mask).count('1')
            self.SHIFTS[square] = 64 - relevant_bits

            # 3) Find a magic number that yields no collisions
            magic = self.find_magic_number(square, relevant_bits, mask)
            self.MAGICS[square] = magic

            # 4) Build the final attack table
            self.generate_attack_table(square, magic, relevant_bits, mask)

        # Save the data to files
        self.save_data()

    def generate_move(self, board: chess.Board, color: bool):
        moves = []

        # Determine the piece type based on self.piece_type
        piece_type = chess.ROOK if self.piece_type == "rook" else chess.BISHOP
        pieces = int(board.pieces(piece_type, color))

        # 2) Occupancy bitboards
        occupancy = int(board.occupied)  # all pieces
        own_occ = int(board.occupied_co[color])  # own pieces only

        for square in BitboardManipulations.bit_scan(pieces):
            # Compute relevant occupancy
            relevant_occ = occupancy & self.MASKS[square]

            # Multiply by magic number and shift
            magic = self.MAGICS[square]
            shift = self.SHIFTS[square]
            index = (relevant_occ * magic) >> shift

            # Look up attacked squares
            attacks = self.ATTACKS[square][index]

            # Remove squares occupied by own pieces
            valid_attacks = attacks & ~own_occ

            # Convert to moves
            for target in BitboardManipulations.bit_scan(valid_attacks):
                moves.append(chess.Move(square, target))
        return moves

    def save_data(self):
        magics_fn = f"{self.piece_type}_magic_numbers.npy",
        attacks_fn = f"{self.piece_type}_attack_table.npz",
        shifts_fn = f"{self.piece_type}_shifts.npy",
        masks_fn = f"{self.piece_type}_masks.npy",

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
        np.save(os.path.join(magics_dir, str(magics_fn)), np.array(self.MAGICS, dtype=np.uint64))

        # Save shifts
        np.save(os.path.join(shifts_dir, str(shifts_fn)), np.array(self.SHIFTS, dtype=np.uint64))

        # Save masks
        np.save(os.path.join(masks_dir, str(masks_fn)), np.array(self.MASKS, dtype=np.uint64))

        # Convert attack tables into a dictionary
        attack_dict = {f"square_{sq}": np.array(self.ATTACKS[sq], dtype=np.uint64) for sq in range(64)}

        # Save attack tables using .npz (compressed format)
        np.savez_compressed(os.path.join(attacks_dir, str(attacks_fn)), **attack_dict)

        print("Data was saved successfully")

    def load_data(self):
        data_files = {
            'magic_numbers': f"data/magics/{self.piece_type}_magic_numbers.npy",
            'attack_table': f"data/magics/{self.piece_type}_attack_table.npz",
            'shifts': f"data/magics/{self.piece_type}_shifts.npy",
            'masks':f"data/magics/{self.piece_type}_masks.npy"}

        # Check for missing files and collect their paths
        missing_files = [str(path) for path in data_files.values() if not os.path.exists(path)]

        if missing_files:
            print("The following required files are missing:")
            for file in missing_files:
                print(f" - {file}")
            ans = input("Do you want to initialize and save the data? [y/N]: ").lower()
            if ans == "y":
                self.initialize_data()
            else:
                return  # Exit the function if the user declines

        # Load magic numbers
        try:
            self.MAGICS = np.load(data_files["magic_numbers"])
            print("Magic numbers loaded")
        except Exception as e:
            print(f"Failed to load magic numbers: {e}")

        # Load shifts
        try:
            self.SHIFTS = np.load(data_files["shifts"])
            print("Shifts loaded")
        except Exception as e:
            print(f"Failed to load shifts: {e}")

        # Load masks
        try:
            self.MASKS = np.load(data_files["masks"])
            print("Masks loaded")
        except Exception as e:
            print(f"Failed to load masks: {e}")

        # Load attack tables from .npz using a context manager
        try:
            with np.load(data_files["attack_table"]) as attack_data:
                self.ATTACKS = {
                    int(key.split("_")[1]): attack_data[key] for key in attack_data.files
                }
            print("Attacks loaded")
        except Exception as e:
            print(f"Failed to load attack tables: {e}")


class Rook(SlidingPiece):
    # Directions: up, down, right, left
    def __init__(self):
        super().__init__("rook", [(1, 0), (-1, 0), (0, 1), (0, -1)])


class Bishop(SlidingPiece):
    # Directions: NE, NW, SE, SW
    def __init__(self):
        super().__init__("bishop", [(1, 1), (1, -1), (-1, 1), (-1, -1)])


# Assuming set_bit and is_bit_set are defined
rook = Rook()
bishop = Bishop()


bishop_attacks = bishop.generate_attacks(27, 0)

def print_bitboard(bitboard: int):
    """Prints a 64-bit bitboard as  8x8 chessboard."""
    bit_string = format(bitboard & 0xFFFFFFFFFFFFFFFF, '064b')
    print(bit_string)

    # Reverse the order to match a chessboard layout
    for rank in range(7, -1, -1):  # Start from rank 7 (top) down to rank 0 (bottom)
        row = bit_string[rank * 8 : (rank + 1) * 8]  # Extract 8 bits per rank
        print(row.replace('0', '.').replace('1', 'X'))  # Replace 1s with 'X' for visibility

