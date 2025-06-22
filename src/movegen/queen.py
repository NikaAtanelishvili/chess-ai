import chess
import numpy as np

from src.sliding_moves.sliding_moves import generate_moves_wrapper


class Queen:
    @staticmethod
    def generate_move(board: chess.Board, color: bool):
        # Determine the piece type based on self.piece_type
        # piece_type = chess.ROOK if self.piece_type == "rook" else chess.BISHOP
        pieces = board.pieces(chess.QUEEN, color)

        # 2) Occupancy bitboards
        occupancy = np.uint64(board.occupied)  # all pieces
        own_occ = np.uint64(board.occupied_co[color])  # own pieces only


        raw_moves_rook = generate_moves_wrapper(pieces, occupancy, own_occ, 0)
        raw_moves_bishop = generate_moves_wrapper(pieces, occupancy, own_occ, 1)


        # Convert raw moves to chess.Move objects and filter for legality
        # from_sq = ((raw_moves >> np.uint16(6)) & np.uint16(0x3F)).astype(np.uint8)
        # to_sq = (raw_moves & np.uint16(0x3F)).astype(np.uint8)

        # 3) Python‐side legality filter
        # legal_moves = []
        # for f, t in zip(from_sq, to_sq):
        #     pass
        #     m = chess.Move(f, t)
        #     if board.is_legal(m):
        #         legal_moves.append(m)
        #
        # return legal_moves
        legal_moves = []
        for move in [*raw_moves_bishop, *raw_moves_rook]:
            from_square = (move >> 6) & 63  # Extract from_square (bits 6-11)
            to_square = move & 63  # Extract to_square (bits 0-5)
            chess_move = chess.Move(from_square, to_square)
            if board.is_legal(chess_move):  # Filter pseudo-legal to legal moves
                legal_moves.append(chess_move)

        return legal_moves