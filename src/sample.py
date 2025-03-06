# import numpy as np
#
# from src.helpers import BitboardManipulations
#
#
# def generate_occupancy_mask(square, directions):
#     mask = 0
#     for direction in directions:
#         mask |= generate_mask_in_direction(square, direction)
#     return mask
#
#
# def generate_mask_in_direction(square, direction):
#     mask = 0
#     dr, df = direction
#     rank, file = divmod(square, 8)
#     r, f = rank + dr, file + df
#
#     while 0 <= r <= 7 and 0 <= f <= 7:
#         sq = r * 8 + f
#         mask = BitboardManipulations.set_bit(mask, sq)
#         r += dr
#         f += df
#
#     return mask
#
# def print_bitboard(bitboard: int):
#     """Prints a 64-bit bitboard as  8x8 chessboard."""
#     bit_string = format(bitboard & 0xFFFFFFFFFFFFFFFF, '064b')
#
#     # Reverse the order to match a chessboard layout
#     for rank in range(7, -1, -1):  # Start from rank 7 (top) down to rank 0 (bottom)
#         row = bit_string[rank * 8 : (rank + 1) * 8]  # Extract 8 bits per rank
#         print(row.replace('0', '.').replace('1', 'X'))  # Replace 1s with 'X' for visibility
#
#
#
# masks = generate_occupancy_mask(6, [(1, 0), (-1, 0), (0, 1), (0, -1)])
# print(masks)
# print_bitboard(masks)
#
# import numpy as np
# import random
# from abc import ABC, abstractmethod
# from typing import List, Tuple
# from tqdm import tqdm
# import chess

import chess
import time


board = chess.Board()
board.set_fen("8/8/8/3Q4/8/8/8/8 w - - 0 1")  # Rook on d5

# Get the square index of the rook (d5)
start = time.time()

rook_square = chess.D5

# Generate all legal moves for the rook
rook_moves = [move for move in board.legal_moves]
end = time.time()
# Print the moves
for move in rook_moves:
    print(move)



print(f"Execution time: {end - start:.6f} seconds")

# sentinel = np.uint64(0xFFFFFFFFFFFFFFFF)
#             used = np.full(size, sentinel, dtype=np.uint64)
# inusde uxsed sentienk

def generate_move(self, board: chess.Board, color: bool):
    moves = []
    piece_type = chess.ROOK if self.piece_type == "rook" else chess.BISHOP
    pieces = board.pieces(piece_type, color)
    occupancy = np.uint64(board.occupied)  # all pieces
    own_occ = np.uint64(board.occupied_co[color])  # own pieces only

    for square in bit_scan(pieces):
        relevant_occ = occupancy & self.MASKS[square]
        magic = self.MAGICS[square]
        shift = self.SHIFTS[square]
        index = (relevant_occ * magic) >> shift
        attacks = self.ATTACKS[square][index]
        valid_attacks = attacks & ~own_occ
        for target in bit_scan(valid_attacks):
            moves.append(target)  # for now

    return moves



