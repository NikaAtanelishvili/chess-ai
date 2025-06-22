import os

import chess
import numpy as np

from src.jumping_moves.jumping_moves import initialize_king_table, generate_king_moves_wrapper

class King:
    ATTACKS = np.zeros(64, dtype=np.uint64)
    OFFSETS = [(1, 0), (-1, 0), (0, 1), (0, -1),
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


    @staticmethod
    def generate_attack_table():
        for square in chess.SQUARES:
            moves = np.uint64(0)

            rank, file = divmod(square, 8)
            for dr, df in King.OFFSETS:
                nr, nf = rank + dr, file + df
                if 0 <= nr < 8 and 0 <= nf < 8:
                    bit_pos = nr * 8 + nf
                    moves |= np.uint64(1) << np.uint64(bit_pos)

            King.ATTACKS[square] = moves


    @staticmethod
    def generate_move( board: chess.Board, color: bool):
        occupancy, own_occ, from_sq, can_ks, can_qs = King.extract_king_state(board, color)

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


    @staticmethod
    def save_data():
        try:
            attacks_fn = f"king_attacks"

            # Define the directory
            attacks_dir = '../data/attacks'

            # Create directories if they don't exist
            os.makedirs(attacks_dir, exist_ok=True)

            np.save(os.path.join(attacks_dir, attacks_fn), King.ATTACKS)

            print(f"King attack tables saved to '{os.path.join(attacks_dir, attacks_fn)}'")
        except Exception as e:
            print(f"Failed to save king attack tables: {e}")


    @staticmethod
    def load_data():
        data_file = f"../data/attacks/king_attacks.npy"

        if not os.path.exists(data_file):
            print("The following required file is missing:")
            print(f" - {data_file}")
            ans = input("Do you want to initialize and save the data? [y/N]: ").lower()
            if ans == "y":
                King.generate_attack_table()
            else:
                return  # Exit the function if the user declines

        try:
            attacks = np.load(data_file).astype(np.uint64)
            initialize_king_table(attacks)
            print("King's Attacks loaded")
        except Exception as e:
            print(f"Failed to load king's attacks: {e}")