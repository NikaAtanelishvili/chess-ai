import os

import chess
import numpy as np

from src.jumping_moves.jumping_moves import initialize_knight_table, generate_knight_moves_wrapper



class Knight:
    ATTACKS = np.zeros(64, dtype=np.uint64)
    OFFSETS = [(2, 1), (2, -1), (-2, 1), (-2, -1),
               (1, 2), (1, -2), (-1, 2), (-1, -2)]

    @staticmethod
    def extract_knight_state(board: chess.Board, color: bool):
        own_occ = np.uint64(board.occupied_co[color])

        pieces = np.uint64(board.pieces(chess.KNIGHT, color))

        return own_occ, pieces


    @staticmethod
    def generate_attack_table():
        for square in chess.SQUARES:
            moves = np.uint64(0)

            rank, file = divmod(square, 8)

            for dr, df in Knight.OFFSETS:
                nr, nf = rank + dr, file + df
                if 0 <= nr < 8 and 0 <= nf < 8:
                    bit_pos = nr * 8 + nf
                    moves |= np.uint64(1) << np.uint64(bit_pos)

            Knight.ATTACKS[square] = moves


    @staticmethod
    def generate_move(board: chess.Board, color: bool):
        own_occ, pieces = Knight.extract_knight_state(board, color)

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


    @staticmethod
    def save_data():
        try:
            attacks_fn = f"knight_attacks"

            # Define the directory
            attacks_dir = '../data/attacks'

            # Create directories if they don't exist
            os.makedirs(attacks_dir, exist_ok=True)

            np.save(os.path.join(attacks_dir, attacks_fn), Knight.ATTACKS)

            print(f"Knight attack tables saved to '{os.path.join(attacks_dir, attacks_fn)}'")
        except Exception as e:
            print(f"Failed to save knight attack tables: {e}")


    @staticmethod
    def load_data():
        data_file = f"../data/attacks/knight_attacks.npy"

        if not os.path.exists(data_file):
            print("The following required file is missing:")
            print(f" - {data_file}")
            ans = input("Do you want to initialize and save the data? [y/N]: ").lower()
            if ans == "y":
                Knight.generate_attack_table()
            else:
                return  # Exit the function if the user declines

        try:
            attacks = np.load(data_file).astype(np.uint64)
            initialize_knight_table(attacks)
            print("Attacks loaded")
        except Exception as e:
            print(f"Failed to load attack table: {e}")
