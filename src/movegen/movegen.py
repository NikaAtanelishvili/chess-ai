import random
import time

import chess

from src.movegen.bishop import Bishop
from src.movegen.king import King
from src.movegen.knight import Knight
from src.movegen.pawn import Pawn
from src.movegen.queen import Queen
from src.movegen.rook import Rook


class MoveGenerator:
    _loaded = False

    piece_generators = [
        King.generate_move,
        Queen.generate_move,
        Rook.generate_move,
        Bishop.generate_move,
        Knight.generate_move,
        Pawn.generate_move,
    ]

    piece_data_loaders = [
        King.load_data,
        Rook.load_data,
        Bishop.load_data,
        Knight.load_data,
        Pawn.load_data,
    ]

    @staticmethod
    def load_all_data():
        if MoveGenerator._loaded:
            return
        for loader in MoveGenerator.piece_data_loaders:
            loader()
        MoveGenerator._loaded = True

    @staticmethod
    def generate_all_moves(board, color):
        all_moves = []
        for gen in MoveGenerator.piece_generators:
            all_moves.extend(gen(board, color))
        return all_moves


if __name__ == "__main__":
    MoveGenerator.load_all_data()
    board = chess.Board()

    start = time.time()
    moves =  MoveGenerator.generate_all_moves(board, board.turn)
    end = time.time()

    start1 = time.time()
    legal_moves = MoveGenerator.generate_all_moves(board, board.turn)
    end1 = time.time()

    print(moves)
    print(len(moves), f'{(end - start):.6f}s')
    print(len(moves), f'{(end1 - start1):.6f}s')

