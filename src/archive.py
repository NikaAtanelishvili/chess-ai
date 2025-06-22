import os
import time
from typing import Tuple

import chess
import numpy as np
from chess import KNIGHT

from src.jumping_moves.jumping_moves import generate_knight_moves_wrapper, generate_pawn_moves_wrapper, \
    initialize_knight_table, initialize_pawn_tables, initialize_king_table, generate_king_moves_wrapper


class King:
    def __init__(self):
        self.ATTACKS = np.zeros(64, dtype=np.uint64)
        self.OFFSETS = [(1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)]


    @staticmethod
    def extract_king_state(board: chess.Board, color: bool):
        occupancy = np.uint64(board.occupied)
        own_occ = np.uint64(board.occupied_co[color])

        king_bb = board.king(color)
        from_sq = (king_bb & -king_bb).bit_length() - 1

        can_ks = board.has_kingside_castling_rights(color)
        can_qs = board.has_queenside_castling_rights(color)

        return occupancy, own_occ, from_sq, can_ks, can_qs


    def generate_attack_table(self):
        for square in chess.SQUARES:
            moves = np.uint64(0)

            rank, file = divmod(square, 8)
            for dr, df in self.OFFSETS:
                nr, nf = rank + dr, file + df
                if 0 <= nr < 8 and 0 <= nf < 8:
                    bit_pos = nr * 8 + nf
                    moves |= np.uint64(1) << np.uint64(bit_pos)

            self.ATTACKS[square] = moves


    def generate_move(self, board: chess.Board, color: bool):
        occupancy, own_occ, from_sq, can_ks, can_qs = self.extract_king_state(board, color)

        raw_moves = generate_king_moves_wrapper(occupancy, own_occ, from_sq, can_ks, can_qs, color)

        # from_sq = ((raw_moves >> 6) & 0x3F).astype(np.uint8)
        # to_sq = (raw_moves & 0x3F).astype(np.uint8)
        #
        # legal_moves = [
        #     chess.Move(f, t)
        #     for f, t in zip(from_sq, to_sq)
        #     if board.is_legal(chess.Move(f, t))
        # ]
        #
        # return legal_moves

        legal_moves = []

        for move in raw_moves:
            from_sq = (move >> 6) & 0x3F
            to_sq = move & 0x3F
            mv = chess.Move(from_sq, to_sq)
            if board.is_legal(mv):
                legal_moves.append(mv)

        return legal_moves


    def save_data(self):
        try:
            attacks_fn = f"king_attacks"

            # Define the directory
            attacks_dir = 'data/attacks'

            # Create directories if they don't exist
            os.makedirs(attacks_dir, exist_ok=True)

            np.save(os.path.join(attacks_dir, attacks_fn), self.ATTACKS)

            print(f"King attack tables saved to '{os.path.join(attacks_dir, attacks_fn)}'")
        except Exception as e:
            print(f"Failed to save king attack tables: {e}")


    def load_data(self):
        data_file = f"data/attacks/king_attacks.npy"

        if not os.path.exists(data_file):
            print("The following required file is missing:")
            print(f" - {data_file}")
            ans = input("Do you want to initialize and save the data? [y/N]: ").lower()
            if ans == "y":
                self.generate_attack_table()
            else:
                return  # Exit the function if the user declines

        try:
            attacks = np.load(data_file).astype(np.uint64)
            initialize_king_table(attacks)
            print("King's Attacks loaded")
        except Exception as e:
            print(f"Failed to load king's attacks: {e}")


class Knight:
    def __init__(self):
        self.ATTACKS = np.zeros(64, dtype=np.uint64)
        self.OFFSETS = [(2, 1), (2, -1), (-2, 1), (-2, -1),
                   (1, 2), (1, -2), (-1, 2), (-1, -2)]


    @staticmethod
    def extract_knight_state(board: chess.Board, color: bool):
        own_occ = np.uint64(board.occupied_co[color])

        pieces = np.uint64(board.pieces(chess.KNIGHT, color))

        return own_occ, pieces


    def generate_attack_table(self):
        for square in chess.SQUARES:
            moves = np.uint64(0)

            rank, file = divmod(square, 8)

            for dr, df in self.OFFSETS:
                nr, nf = rank + dr, file + df
                if 0 <= nr < 8 and 0 <= nf < 8:
                    bit_pos = nr * 8 + nf
                    moves |= np.uint64(1) << np.uint64(bit_pos)

            self.ATTACKS[square] = moves


    def generate_move(self, board: chess.Board, color: bool):
        own_occ, pieces = self.extract_knight_state(board, color)

        raw_moves = generate_knight_moves_wrapper(pieces, own_occ, color)
        #
        # from_sq = ((raw_moves >> 6) & 0x3F).astype(np.uint8)
        # to_sq = (raw_moves & 0x3F).astype(np.uint8)
        #
        # legal_moves = [
        #     mv for f, t in zip(from_sq, to_sq)
        #     if board.is_legal(mv := chess.Move(f, t))
        # ]

        legal_moves = []

        for move in raw_moves:
            from_sq = (move >> 6) & 0x3F
            to_sq = move & 0x3F
            mv = chess.Move(from_sq, to_sq)
            if board.is_legal(mv):
                legal_moves.append(mv)

        return legal_moves

    def save_data(self):
        try:
            attacks_fn = f"knight_attacks"

            # Define the directory
            attacks_dir = 'data/attacks'

            # Create directories if they don't exist
            os.makedirs(attacks_dir, exist_ok=True)

            np.save(os.path.join(attacks_dir, attacks_fn), self.ATTACKS)

            print(f"Knight attack tables saved to '{os.path.join(attacks_dir, attacks_fn)}'")
        except Exception as e:
            print(f"Failed to save knight attack tables: {e}")


    def load_data(self):
        data_file = f"data/attacks/knight_attacks.npy"

        if not os.path.exists(data_file):
            print("The following required file is missing:")
            print(f" - {data_file}")
            ans = input("Do you want to initialize and save the data? [y/N]: ").lower()
            if ans == "y":
                self.generate_attack_table()
            else:
                return  # Exit the function if the user declines

        try:
            attacks = np.load(data_file).astype(np.uint64)
            initialize_knight_table(attacks)
            print("Attacks loaded")
        except Exception as e:
            print(f"Failed to load attack table: {e}")


class Pawn:
    def __init__(self):
        self.ATTACKS_WHITE = np.zeros(64, dtype=np.uint64)
        self.ATTACKS_BLACK = np.zeros(64, dtype=np.uint64)


    @staticmethod
    def extract_pawn_state(board: chess.Board, color: bool) -> Tuple[np.uint64, np.uint64, np.uint64, int]:

        occupancy = np.uint64(board.occupied)

        pieces = np.uint64(board.pieces(chess.PAWN, color))  # occupancy of pawns

        opp_occ = np.uint64(board.occupied_co[not color])  # occupancy board of enemy time

        ep_sq = board.ep_square if board.ep_square is not None else -1 # en-passant square

        return occupancy, pieces, opp_occ, ep_sq


    def generate_attack_table_white(self):
        for square in chess.SQUARES:

            moves = np.uint64(0)

            rank, file = divmod(square, 8)

            if rank < 7:  # Can always push one square if not on last rank.
                if file > 0:
                    moves |= np.uint64(1) << np.uint64(square + 7) # Left capture
                if file < 7:
                    moves |= np.uint64(1) << np.uint64(square + 9) # Right capture

            self.ATTACKS_WHITE[square] = moves


    def generate_attack_table_black(self):
        for square in chess.SQUARES:

            moves = np.uint64(0)

            rank, file = divmod(square, 8)

            if rank > 0:
                if file > 0:
                    moves |= np.uint64(1) << np.uint64(square - 9)
                if file < 7:
                    moves |= np.uint64(1) << np.uint64(square - 7)

            self.ATTACKS_BLACK[square] = moves


    def generate_move(self, board: chess.Board, color: bool):
        occupancy, pieces, opp_occ, ep_sq = self.extract_pawn_state(board, color)

        raw_moves = generate_pawn_moves_wrapper(pieces, occupancy, opp_occ, color, ep_sq)

        legal_moves = []
        for move in raw_moves:
            from_sq = (move >> 6) & 0x3F
            to_sq = move & 0x3F
            promo = (move >> 12) & 0x7

            # reconstruct python-chess Move
            mv = chess.Move(from_sq, to_sq,
                            promotion=(promo if promo else None))
            if board.is_legal(mv):
                legal_moves.append(mv)
        return legal_moves


    def save_data(self):
        try:
            attacks_fn = f"pawn_attacks"

            # Define the directory
            attacks_dir = 'data/attacks'

            # Create directories if they don't exist
            os.makedirs(attacks_dir, exist_ok=True)

            attack_dict = {
                'white': self.ATTACKS_WHITE,
                'black': self.ATTACKS_BLACK
            }

            np.savez_compressed(os.path.join(attacks_dir, attacks_fn), **attack_dict)

            print(f"Pawn attack tables saved to '{os.path.join(attacks_dir, attacks_fn)}'")

        except Exception as e:
            print(f"Failed to save pawn attack tables: {e}")


    def load_data(self):
        data_file = f"data/attacks/pawn_attacks.npz"

        if not os.path.exists(data_file):
            print("The following required file is missing:")
            print(f" - {data_file}")
            ans = input("Do you want to initialize and save the data? [y/N]: ").lower()
            if ans == "y":
                self.generate_attack_table_white()
                self.generate_attack_table_black()
            else:
                return  # Exit the function if the user declines

        try:
            attacks = np.load(data_file)
            attacks_white = attacks['white']
            attacks_black = attacks['black']
            initialize_pawn_tables(attacks_white, attacks_black)
            print("Attacks loaded")
        except Exception as e:
            print(f"Failed to load attack table: {e}")



def print_bitboard(bitboard):
    """Prints a 64-bit bitboard as  8x8 chessboard."""
    bit_string = format(bitboard & 0xFFFFFFFFFFFFFFFF, '064b')
    print(bit_string)

    # Reverse the order to match a chessboard layout
    for rank in range(7, -1, -1):  # Start from rank 7 (top) down to rank 0 (bottom)
        row = bit_string[rank * 8 : (rank + 1) * 8]  # Extract 8 bits per rank
        print(row.replace('0', '.').replace('1', 'X'))  # Replace 1s with 'X' for visibility

if __name__ == "__main__":
    board = chess.Board()

    # Clear pieces between the king and rook
    # board.remove_piece_at(chess.F1)
    # board.remove_piece_at(chess.G1)
    # board.set_piece_at(chess.F3, chess.Piece(chess.KNIGHT, chess.BLACK))

    # print(KING_MOVES)
    # print_bitboard(KING_MOVES[60])

    # king = King()
    # king.generate_attack_table()

    # pawn = Pawn()
    # pawn.generate_attack_table_white()
    # pawn.generate_attack_table_black()
    # pawn.save_data()
    #
    # king = King()
    # king.generate_attack_table()
    # king.save_data()
    #
    # knight = Knight()
    # knight.generate_attack_table()
    # knight.save_data()

    # pawn = Pawn()
    # pawn.load_data()
    # moves = pawn.generate_move(board, True)

    knight = Pawn()
    knight.load_data()
    # moves = knight.generate_move(board, True)

    start5 = time.time()
    moves5 = knight.generate_move(board, True)
    end5 = time.time()

    start6 = time.time()
    moves6 = knight.generate_move(board, True)
    end6 = time.time()

    start7 = time.time()
    moves7 = knight.generate_move(board, True)
    end7 = time.time()

    start8 = time.time()
    moves8 = knight.generate_move(board, True)
    end8 = time.time()

    #
    # # bishop_attacks = bishop.generate_attacks(27, 0)
    # for move in moves:
    #     print(move)
    #
    #
    print(f'{end5 - start5:.10f} seconds')
    print(f'{end6 - start6:.10f} seconds')
    print(f'{end7 - start7:.10f} seconds')
    print(f'{end8 - start8:.10f} seconds')
# 0.00014-8
    #
    # print(moves)
    #
    # for move in pawn.ATTACKS_WHITE:
    #     print_bitboard(move)
    # board = chess.Board()
    # Generate moves for White using our bitboard method.
    # white_moves = generate_moves_bitboard(board, chess.WHITE)
    # for move in white_moves:
    #     print(move.uci())
