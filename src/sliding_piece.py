import os
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Generator

import chess
import numpy as np
from tqdm import tqdm

from helpers import BitboardManipulations

PRINT_ATTACKS_INIT_DETAILS = True
PRINT_MAGICS_INIT_DETAILS = True
PRINT_SHIFTS_INIT_DETAILS = True
PRINT_MASKS_INIT_DETAILS = True


class SlidingPiece(ABC):
    def __init__(self, piece_type: str, directions: List[Tuple[int, int]]):
        self.directions = directions
        self.piece_type = piece_type

        self.MAGICS = np.zeros(64, dtype=np.uint64)
        self.SHIFTS = np.zeros(64, dtype=np.uint8)
        self.MASKS = np.zeros(64, dtype=np.uint64)
        self.ATTACKS = [np.zeros(1, dtype=np.uint64) for _ in range(64)] # Each will later be a numpy array of np.uint64


    @staticmethod
    def _generate_attacks_in_direction(square: int, blockers: np.uint64, direction: Tuple[int, int]) -> np.uint64:
        attacks = np.uint64(0)
        dr, df = direction
        rank, file = divmod(square, 8)
        r, f = rank + dr, file + df

        while 0 <= r < 8 and 0 <= f < 8:
            sq = r * 8 + f
            # Assuming BitboardManipulations.set_bit is updated to work with np.uint64
            attacks = BitboardManipulations.set_bit(attacks, sq)
            if BitboardManipulations.is_bit_set(blockers, sq):
                break
            r += dr
            f += df

        return attacks


    @staticmethod
    def generate_occupancy_subsets(mask: np.uint64) -> Generator[Tuple[int, np.uint64], None, None]:
        """
        Generates all possible occupancies (subsets) of a given bitboard mask
        Each subset represents a different combination of occupied squares within the mask.

        mask: A bitboard (64-bit integer) representing a set of squares
        (e.g., squares that could contain blocking pieces for a rook)
        """
        i = 0
        sub = np.uint64(0)
        while True:
            yield i, sub
            i+=1
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
            if sub == np.uint64(0):
                break
            # Think of this process like counting in binary


    @abstractmethod
    def generate_occupancy_mask(self, square: int) -> np.uint64:
        raise NotImplementedError("This method should be overridden in subclasses.")


    def generate_attacks(self, square: int, blockers: np.uint64) -> np.uint64:
        attacks = np.uint64(0)
        for direction in self.directions:
            attacks |= self._generate_attacks_in_direction(square, blockers, direction)
        return attacks


    def find_magic_number(self, square: int, relevant_bits: int, mask: np.uint64) -> np.uint64:

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

        occupancies = np.zeros(size, dtype=np.uint64)
        attacks_table = np.zeros(size, dtype=np.uint64)

        # Generate all possible occupancy subsets and their attacks
        for i, occ in self.generate_occupancy_subsets(mask):
            occupancies[i] = occ
            attacks_table[i] = self.generate_attacks(square, occ)

        # Try random 64-bit numbers until a collision-free magic is found
        while True:
            magic = (np.uint64(random.getrandbits(64)) &
                     np.uint64(random.getrandbits(64)) &
                     np.uint64(random.getrandbits(64)))

            # "good" magic number needs high bit entropy (spread-out bits for better indexing)
            if magic & np.uint64(0xFF00000000000000) == np.uint64(0): # isolates the top 8 bits and checks if the top 8 bits of magic are all zeros
                continue

            used = np.full(size, -1, dtype=np.int64)
            fail = False

            # Test the magic number for collisions
            for i, occ in enumerate(occupancies):
                index = ((occ * magic) & np.uint64(0xFFFFFFFFFFFFFFFF)) >> np.uint64(64 - relevant_bits)
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
                if used[index] == np.uint64(-1):
                    used[index] = attacks_table[i]
                elif used[index] != attacks_table[i]:
                    fail = True
                    break

            if not fail:
                return magic


    def generate_attack_table(self, square: int, magic: np.uint64, relevant_bits: int, mask: np.uint64) -> np.ndarray:
        """
        Args:
            square (int): The square index (0-63).
            magic (int): The magic number for this square.
            relevant_bits (int): Number of bits in the occupancy mask.
            mask (int): Relevant occupancy mask bitboard.
        """
        size = 1 << relevant_bits
        attacks_array = np.zeros(size, dtype=np.uint64)
        # Precompute occupancy -> index -> attacks
        for _, occ in self.generate_occupancy_subsets(mask):
            index = ((occ * magic) & np.uint64(0xFFFFFFFFFFFFFFFF)) >> np.uint64(64 - relevant_bits)
            # Compute the actual rook attacks given 'occ'
            attacks_array[index] = self.generate_attacks(square, occ)

        return attacks_array


    def initialize_data(self) -> None:
        """
        Initialize the piece's data: magic numbers, attacks table, shifts, and masks.
        """
        for square in tqdm(range(64), desc='Initializing the data: MAGICS, ATTACKS, SHIFTS, MASKS'):
            # 1) Compute the relevant mask for this square
            mask = self.generate_occupancy_mask(square)
            self.MASKS[square] = mask

            if PRINT_MASKS_INIT_DETAILS:
                print(f'Occupancy_mask for square №{square}:', end='\n')
                print_bitboard(mask)

            # 2) Count how many bits are set -> relevant_bits
            relevant_bits = bin(mask).count('1')
            self.SHIFTS[square] = np.uint8(64 - relevant_bits)

            if PRINT_SHIFTS_INIT_DETAILS:
                print(f'Shift for square №{square}: {64-relevant_bits}', end='\n')

            # 3) Find a magic number that yields no collisions
            magic = self.find_magic_number(square, relevant_bits, mask)
            self.MAGICS[square] = magic
            if PRINT_MAGICS_INIT_DETAILS:
                    print(f'Magic number was founded for square №{square} ✅ | magic number: {magic}', end='\n')

            # 4) Build the final attack table
            attacks = self.generate_attack_table(square, magic, relevant_bits, mask)
            self.ATTACKS[square] = attacks
            if PRINT_ATTACKS_INIT_DETAILS:
                print(f'Attack table was created for square №{square} | size: {len(attacks)}', end='\n')
                print_bitboard(attacks[0])

            # Print progress
            progress = (square + 1) / 64 * 100
            print(f"Progress: {progress:.2f}%", end="\r")  # Overwrites the same line

        # Save the data to files
        self.save_data()


    def generate_move(self, board: chess.Board, color: bool):
        moves = []

        # Determine the piece type based on self.piece_type
        piece_type = chess.ROOK if self.piece_type == "rook" else chess.BISHOP
        pieces = np.uint64(int(board.pieces(piece_type, color)))

        # 2) Occupancy bitboards
        occupancy = np.uint64(int(board.occupied))  # all pieces
        own_occ = np.uint64(int(board.occupied_co[color]))  # own pieces only

        for square in BitboardManipulations.bit_scan(pieces):
            # Compute relevant occupancy
            relevant_occ = occupancy & self.MASKS[square]

            # Multiply by magic number and shift
            magic = self.MAGICS[square]
            shift = self.SHIFTS[square]
            index = (relevant_occ * magic) >> np.uint64(shift)

            # Look up attacked squares
            attacks = self.ATTACKS[square][index]
            print_bitboard(attacks)
            # Remove squares occupied by own pieces
            valid_attacks = attacks & ~own_occ

            # Convert to moves
            for target in BitboardManipulations.bit_scan(valid_attacks):
                moves.append(chess.Move(square, target)) # for now
        return moves


    def save_data(self):
        magics_fn = f"{self.piece_type}_magics"
        attacks_fn = f"{self.piece_type}_attacks"
        shifts_fn = f"{self.piece_type}_shifts"
        masks_fn = f"{self.piece_type}_masks"

        # Define the directories for each type of data
        magics_dir = "data/magics"
        attacks_dir = 'data/attacks'
        shifts_dir = 'data/shifts'
        masks_dir = 'data/masks'

        # Create directories if they don't exist
        os.makedirs(magics_dir, exist_ok=True)
        os.makedirs(attacks_dir, exist_ok=True)
        os.makedirs(shifts_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        # Save magic numbers
        np.save(os.path.join(magics_dir, magics_fn), np.array(self.MAGICS, dtype=np.uint64))

        # Save shifts
        np.save(os.path.join(shifts_dir, shifts_fn), np.array(self.SHIFTS, dtype=np.uint8))

        # Save masks
        np.save(os.path.join(masks_dir, masks_fn), np.array(self.MASKS, dtype=np.uint64))

        # Convert attack tables into a dictionary
        attack_dict = {f"square_{sq}": np.array(self.ATTACKS[sq], dtype=np.uint64) for sq in range(64)}

        # Save attack tables using .npz (compressed format)
        np.savez_compressed(os.path.join(attacks_dir, attacks_fn), **attack_dict)

        print("Data was saved successfully")


    def load_data(self):
        data_files = {
            'magic_numbers': f"data/magics/{self.piece_type}_magics.npy",
            'attack_table': f"data/attacks/{self.piece_type}_attacks.npz",
            'shifts': f"data/shifts/{self.piece_type}_shifts.npy",
            'masks':f"data/masks/{self.piece_type}_masks.npy"}

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
            self.MAGICS = np.load(data_files["magic_numbers"]).astype(np.uint64)
            print("Magic numbers loaded")
        except Exception as e:
            print(f"Failed to load magic numbers: {e}")

        # Load shifts
        try:
            self.SHIFTS = np.load(data_files["shifts"]).astype(np.uint8)
            print("Shifts loaded")
        except Exception as e:
            print(f"Failed to load shifts: {e}")

        # Load masks
        try:
            self.MASKS = np.load(data_files["masks"]).astype(np.uint64)
            print("Masks loaded")
        except Exception as e:
            print(f"Failed to load masks: {e}")

        # Load attack tables from .npz using a context manager
        try:
            with np.load(data_files["attack_table"]) as attack_data:
                self.ATTACKS = {
                    int(key.split("_")[1]): attack_data[key].astype(np.uint64) for key in attack_data.files
                }
            print("Attacks loaded")
            print(type(self.ATTACKS[0][0]))
        except Exception as e:
            print(f"Failed to load attack tables: {e}")


class Rook(SlidingPiece):
    # Directions: up, down, right, left
    def __init__(self):
        super().__init__("rook", [(1, 0), (-1, 0), (0, 1), (0, -1)])


    def generate_occupancy_mask(self, square):
        """
        For a rook on 'square'(0-64), compute the 'relevant occupancy mask'
        (the squares that can block the rook).

        -The edge squares don’t contribute to attack variations because
        the rook would always move up to the board’s edge.
        """
        mask = np.uint64(0)
        rank, file = divmod(square, 8)

        # Up (exclude final rank edge)
        for r in range(rank + 1, 7):
            sq = r * 8 + file
            mask = BitboardManipulations.set_bit(mask, sq)

        # Down (exclude final rank edge)
        for r in range(rank - 1, 0, -1):
            sq = r * 8 + file
            mask = BitboardManipulations.set_bit(mask, sq)

        # Right (exclude final file edge)
        for f in range(file + 1, 7):
            sq = rank * 8 + f
            mask = BitboardManipulations.set_bit(mask, sq)

        # Left (exclude final file edge)
        for f in range(file - 1, 0, -1):
            sq = rank * 8 + f
            mask = BitboardManipulations.set_bit(mask, sq)

        return mask


class Bishop(SlidingPiece):
    # Directions: NE, NW, SE, SW
    def __init__(self):
        super().__init__("bishop", [(1, 1), (1, -1), (-1, 1), (-1, -1)])


    def generate_occupancy_mask(self, square):
        """
        For a bishop on 'square'(0-64), compute the 'relevant occupancy mask'
        (the squares that can block the bishop).

        -The edge squares don’t contribute to attack variations because
        the bishop would always move up to the board’s edge.
        """
        mask = np.uint64(0)
        rank, file = divmod(square, 8)

        # Northeast (↗️)
        r, f = rank + 1, file + 1
        while r < 7 and f < 7:
            sq = r * 8 + f
            mask = BitboardManipulations.set_bit(mask, sq)
            r += 1
            f += 1

        # Northwest (↖️)
        r, f = rank + 1, file - 1
        while r < 7 and f > 0:
            sq = r * 8 + f
            mask = BitboardManipulations.set_bit(mask, sq)
            r += 1
            f -= 1

        # Southeast (↘️)
        r, f = rank - 1, file + 1
        while r > 0 and f < 7:
            sq = r * 8 + f
            mask = BitboardManipulations.set_bit(mask, sq)
            r -= 1
            f += 1

        # Southwest (↙️)
        r, f = rank - 1, file - 1
        while r > 0 and f > 0:
            sq = r * 8 + f
            mask = BitboardManipulations.set_bit(mask, sq)
            r -= 1
            f -= 1

        return mask


def print_bitboard(bitboard):
    """Prints a 64-bit bitboard as  8x8 chessboard."""
    bit_string = format(bitboard & 0xFFFFFFFFFFFFFFFF, '064b')
    print(bit_string)

    # Reverse the order to match a chessboard layout
    for rank in range(7, -1, -1):  # Start from rank 7 (top) down to rank 0 (bottom)
        row = bit_string[rank * 8 : (rank + 1) * 8]  # Extract 8 bits per rank
        print(row.replace('0', '.').replace('1', 'X'))  # Replace 1s with 'X' for visibility



if __name__ == '__main__':
    rook = Rook()
    # bishop = Bishop()
    # bishop.load_data()
    PRINT_DETAILS = True
    rook.load_data()
    board = chess.Board()
    board.set_fen("8/8/8/3R4/8/8/8/8 w - - 0 1")  # Rook on d5
    moves = rook.generate_move(board, True)
    # bishop_attacks = bishop.generate_attacks(27, 0)
    for move in moves:
        print(move)

# def generate_occupancy_mask(self, square: int) -> int:
#     """
#      For a sliding piece on 'square'(0-64), compute the 'relevant occupancy mask'
#      (the squares that can block the rook).
#
#      -The edge squares don’t contribute to attack variations because
#      the rook would always move up to the board’s edge.
#      """
#     mask = 0
#     for direction in self.directions:
#         mask |= self._generate_mask_in_direction(square, direction)
#     return mask
    # @staticmethod
    # def _generate_mask_in_direction(square: int, direction: Tuple[int, int]) -> int:
    #     mask = 0
    #     dr, df = direction
    #     rank, file = divmod(square, 8)
    #     r, f = rank + dr, file + df
    #
    #     while 0 <= r <= 7 and 0 <= f <= 7:
    #         sq = r * 8 + f
    #         if r == 0 or r == 7 or f == 0 or f == 7:
    #             break
    #         mask = BitboardManipulations.set_bit(mask, sq)
    #         r += dr
    #         f += df
    #
    #     return mask