import os
from typing import Tuple

import chess
import numpy as np

from src.jumping_moves.jumping_moves import initialize_pawn_tables, generate_pawn_moves_wrapper



class Pawn:
    ATTACKS_WHITE = np.zeros(64, dtype=np.uint64)
    ATTACKS_BLACK = np.zeros(64, dtype=np.uint64)

    @staticmethod
    def extract_pawn_state(board: chess.Board, color: bool) -> Tuple[np.uint64, np.uint64, np.uint64, int]:

        occupancy = np.uint64(board.occupied)

        pieces = np.uint64(board.pieces(chess.PAWN, color))  # occupancy of pawns

        opp_occ = np.uint64(board.occupied_co[not color])  # occupancy board of enemy time

        ep_sq = board.ep_square if board.ep_square is not None else -1 # en-passant square

        return occupancy, pieces, opp_occ, ep_sq

    @staticmethod
    def generate_attack_table_white():
        for square in chess.SQUARES:

            moves = np.uint64(0)

            rank, file = divmod(square, 8)

            if rank < 7:  # Can always push one square if not on last rank.
                if file > 0:
                    moves |= np.uint64(1) << np.uint64(square + 7) # Left capture
                if file < 7:
                    moves |= np.uint64(1) << np.uint64(square + 9) # Right capture

            Pawn.ATTACKS_WHITE[square] = moves

    @staticmethod
    def generate_attack_table_black():
        for square in chess.SQUARES:

            moves = np.uint64(0)

            rank, file = divmod(square, 8)

            if rank > 0:
                if file > 0:
                    moves |= np.uint64(1) << np.uint64(square - 9)
                if file < 7:
                    moves |= np.uint64(1) << np.uint64(square - 7)

            Pawn.ATTACKS_BLACK[square] = moves

    @staticmethod
    def generate_move(board: chess.Board, color: bool):
        occupancy, pieces, opp_occ, ep_sq = Pawn.extract_pawn_state(board, color)

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

    @staticmethod
    def save_data():
        try:
            attacks_fn = f"pawn_attacks"

            # Define the directory
            attacks_dir = '../data/attacks'

            # Create directories if they don't exist
            os.makedirs(attacks_dir, exist_ok=True)

            attack_dict = {
                'white': Pawn.ATTACKS_WHITE,
                'black': Pawn.ATTACKS_BLACK
            }

            np.savez_compressed(os.path.join(attacks_dir, attacks_fn), **attack_dict)

            print(f"Pawn attack tables saved to '{os.path.join(attacks_dir, attacks_fn)}'")

        except Exception as e:
            print(f"Failed to save pawn attack tables: {e}")

    @staticmethod
    def load_data():
        data_file = f"../data/attacks/pawn_attacks.npz"

        if not os.path.exists(data_file):
            print("The following required file is missing:")
            print(f" - {data_file}")
            ans = input("Do you want to initialize and save the data? [y/N]: ").lower()
            if ans == "y":
                Pawn.generate_attack_table_white()
                Pawn.generate_attack_table_black()
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