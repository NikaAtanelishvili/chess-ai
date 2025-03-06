import chess

def move_generator (board_state: chess.Board):
    return [move.uci() for move in board_state.legal_moves]


if __name__ == "__main__":
    board = chess.Board()
    moves = move_generator(board)
    print(moves)
