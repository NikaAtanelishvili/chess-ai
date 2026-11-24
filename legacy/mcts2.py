#!/usr/bin/env python3
from collections import deque

import chess
import numpy as np
import math
import random

from chess.polyglot import zobrist_hash

from src.cnn import CNN
from src.cnn_inference import CNNInference
from src.utils import board_to_tensor, create_board_history


def mirror_move(move: chess.Move) -> chess.Move:
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
        drop=move.drop,  # for variants that support drops
    )


def canonicalize_board(board: chess.Board, ai_color: chess.Color) -> chess.Board:

    if ai_color == chess.BLACK:
        canon = board.mirror()  # new Board with pieces, castling, ep, and turn flipped
    else:
        canon = board.copy()  # new Board identical to the original

    # Now ensure it’s White to move (root call should always be on AI’s turn)
    canon.turn = chess.WHITE
    return canon


def canonicalize_move(move: chess.Move, ai_color: chess.Color) -> chess.Move:
    if ai_color == chess.BLACK:
        return mirror_move(move)
    return move


def uncanonicalize_move(move: chess.Move, ai_color: chess.Color) -> chess.Move:
    """
    Map a move from the canonical board back into the original
    board’s coordinate system.
    """
    if ai_color == chess.BLACK:
        return mirror_move(move)
    return move

# deltas for actions
DELTAS = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
KNIGHT_DELTAS = [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]
ACTIONS = [] # (dx, dy, is_promotion, promotion_piece)

# Sliding moves: 8 directions × distances 1–7
for dx, dy in DELTAS:
    for dist in range(1, 8):
        ACTIONS.append((dx * dist, dy * dist, False, None))

# Knight jumps: 8
for dx, dy in KNIGHT_DELTAS:
    ACTIONS.append((dx, dy, False, None))

# UNDERPROM_DIRS = [(0,1),(1,1),(-1,1)]  # relative to white; board is
# new
UNDERPROM_DIRS = [(1,0),(1,1),(1,-1)]

for promo_piece in (chess.ROOK, chess.BISHOP, chess.KNIGHT):
    for dx, dy in UNDERPROM_DIRS:
        ACTIONS.append((dx, dy, True, promo_piece))

assert len(ACTIONS) == 73, f"Expected 73 actions, got {len(ACTIONS)}"

POLICY_SIZE = 64 * len(ACTIONS)  # 4672

# encoding
def move_to_index(move: chess.Move, board: chess.Board) -> int:
    # Mirror into white‐to‐move perspective if needed
    if board.turn == chess.BLACK:
        from_sq = chess.square_mirror(move.from_square)
        to_sq   = chess.square_mirror(move.to_square)
        promotion = move.promotion
    else:
        from_sq, to_sq = move.from_square, move.to_square
        promotion      = move.promotion

    fr_r, fr_c = divmod(from_sq, 8)
    to_r, to_c = divmod(to_sq,   8)
    dx, dy     = to_r - fr_r, to_c - fr_c

    # Only underpromos get promo_flag=True
    is_promo = (promotion is not None and promotion != chess.QUEEN)

    for aid, (adx, ady, promo_flag, promo_piece) in enumerate(ACTIONS):
        if (adx, ady, promo_flag) == (dx, dy, is_promo):
            # match the right piece for underpromos
            if (not promo_flag) or (promotion == promo_piece):
                return from_sq * len(ACTIONS) + aid

    raise ValueError(f"Illegal move for ACTIONS: {move.uci()}")


# decoding
def index_to_move(index: int, turn: chess.Color) -> chess.Move:
    n        = len(ACTIONS)
    from_sq  = index // n
    aid      = index % n
    dx, dy, promo_flag, promo_piece = ACTIONS[aid]

    fr_r, fr_c = divmod(from_sq, 8)
    to_r, to_c = fr_r + dx, fr_c + dy

    # off‐board?
    if not (0 <= to_r < 8 and 0 <= to_c < 8):
        return None

    to_sq = to_r * 8 + to_c

    # start with underpromo if flagged
    promotion = promo_piece if promo_flag else None

    # detect *queen* promotions: a one‐step pawn move onto rank 7 (canonical)
    # (dx==1, abs(dy)<=1, started on rank 6)
    if not promo_flag and fr_r == 6 and dx == 1 and abs(dy) <= 1:
        promotion = chess.QUEEN

    # build the canonical‐white move
    mv = chess.Move(from_sq, to_sq, promotion=promotion)

    # mirror back for Black
    if turn == chess.BLACK:
        mv = chess.Move(
            chess.square_mirror(mv.from_square),
            chess.square_mirror(mv.to_square),
            promotion=mv.promotion
        )
    return mv
# Action encoding: 8×8×73 directional move representation
# DELTAS = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
# KNIGHT_DELTAS = [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]
# ACTIONS = [] # (dx, dy, is_promotion, promotion_piece)
#
# # Sliding moves: 8 directions × distances 1–7
# for dx, dy in DELTAS:
#     for dist in range(1, 8):
#         ACTIONS.append((dx * dist, dy * dist, False, None))
#
# # Knight jumps: 8
# for dx, dy in KNIGHT_DELTAS:
#     ACTIONS.append((dx, dy, False, None))
#
# UNDERPROM_DIRS = [(0,1),(1,1),(-1,1)]  # relative to white; board is canonicalized
# for promo_piece in (chess.ROOK, chess.BISHOP, chess.KNIGHT):
#     for dx, dy in UNDERPROM_DIRS:
#         ACTIONS.append((dx, dy, True, promo_piece))
#
# assert len(ACTIONS) == 73, f"Expected 73 actions, got {len(ACTIONS)}"
#
# POLICY_SIZE = 64 * len(ACTIONS)  # 4672
#
# # I am tyring to use directional move encodoing. Promotional moves in ACTIONS include 9 different elemment(rook, bishop, knight*3 direction).
# # I know thats the standard because canonicalization is basically nessesay all arount my MCTS-CNN. I am having problems with promotion encoding and decoding
# # for example: move_to_index fails, because c2b1r is illigle move. Find the problem, and fix it
#
# # Move-index conversions
# def index_to_move(index: int, turn: chess.Color) -> chess.Move:
#     n = len(ACTIONS)
#     from_sq = index // n
#     aid     = index % n
#     dx, dy, promo_flag, promo_piece = ACTIONS[aid]
#
#     fr_r, fr_c = divmod(from_sq, 8)
#     to_r, to_c = fr_r + dx, fr_c + dy
#     to_sq = to_r * 8 + to_c
#     promotion = promo_piece if promo_flag else None
#
#     # build the canonical (white-to-move) move
#     mv = chess.Move(from_sq, to_sq, promotion=promotion)
#
#     # if we originally mirrored for Black, mirror back:
#     if turn == chess.BLACK:
#         mv = chess.Move(
#             chess.square_mirror(mv.from_square),
#             chess.square_mirror(mv.to_square),
#             promotion=mv.promotion
#         )
#     return mv
#
#
# def move_to_index(move: chess.Move, board: chess.Board) -> int:
#     # 1) mirror squares if Black
#     if not board.turn:
#         orig_from, orig_to = move.from_square, move.to_square
#         from_sq = chess.square_mirror(orig_from)
#         to_sq   = chess.square_mirror(orig_to)
#         promotion = move.promotion
#     else:
#         from_sq, to_sq = move.from_square, move.to_square
#         promotion = move.promotion
#
#     fr_r, fr_c = divmod(from_sq, 8)
#     to_r, to_c = divmod(to_sq, 8)
#     dx, dy = to_r - fr_r, to_c - fr_c
#
#     # 2) only underpromotions are treated as promo=True
#     is_promo = (promotion is not None
#                 and promotion != chess.QUEEN)
#
#     for aid, (adx, ady, promo_flag, promo_piece) in enumerate(ACTIONS):
#         if (adx, ady, promo_flag) == (dx, dy, is_promo):
#             # queen promotions now go into the sliding moves
#             if not promo_flag or promotion == promo_piece:
#                 # use the *mirrored* from_sq for indexing
#                 return from_sq * len(ACTIONS) + aid
#
#     raise ValueError(f"Illegal move for ACTIONS: {move}")

class Node:
    def __init__(self, key: int, parent=None):
        self.key = key                                      # Zobrist hash of position
        self.parent = parent                                # Parent node
        self.children = {}                                  # action_index -> Node
        # self.P = np.zeros(POLICY_SIZE, dtype=np.float32)    # Prior policy P(s,a) | Prior probability vector
        self.P = 0.0
        self.N = 0                                          # Visit count
        self.W = 0.0                                        # Total value
        self.Q = 0.0                                        # Mean value W/N
        self.is_expanded = False

class TranspositionTable:
    def __init__(self):
        self.table = {}             # key -> Node

    def get(self, key: int):
        return self.table.get(key)

    def put(self, node: Node):
        self.table[node.key] = node

def call_count(func):
    def wrapper(self, *args, **kwargs):
        wrapper.call_count +=1
        return func(self, *args, **kwargs)
    wrapper.call_count = 0
    return wrapper

class ChessAI:
    def __init__(self, model, color=chess.WHITE, sims=800, c_puct=1.0):
        self.model = model
        self.color = color
        self.sims = sims
        self.c_puct = c_puct
        self.tt = TranspositionTable()
        self.cache = {}             # key:int -> (v, policy)

    def select_move(self, board: chess.Board, use_dirichlet=False, temperature=0.0) -> chess.Move:
        canon = canonicalize_board(board, self.color)
        root_key = zobrist_hash(canon)
        root = self.tt.get(root_key) or Node(root_key)
        root.parent = None

        self.tt.put(root)

        # Build history everytime select_move is called
        # game_history = create_board_history(board)
        game_history = self._create_canonical_history(board)

        # Initial evaluation
        v_root, policy_root = self._evaluate(canon, game_history)


        if use_dirichlet:
            policy_root = self._add_dirichlet_noise(policy_root)

        root.P = policy_root

        # expansion
        self._expand(root, canon)

        # MCTS simulations
        for _ in range(self.sims):
            board_copy = canon.copy()
            self._simulate(root, board_copy, game_history)
        
        # NOTE: parallelism is guaranteed in this specific case, but only in modern Python versions.
        # As of Python 3.7+, the built-in dict maintains insertion order as an official language feature.
        # So, root.children.keys() and root.children.values() will return items in corresponding, consistent order.
        visits = np.array([node.N for node in root.children.values()], dtype=np.float32)
        actions = list(root.children.keys())

        if temperature == 0.0:
            best = np.argmax(visits)
        else:
            # Stochastic Sampling (𝜏>0)
            # 𝜏=0 → fully greedy: always the max-visit move ( the best exploitation).
            # 𝜏 = 1 → raw visits: sample in proportion to exact visit counts.
            # 𝜏 > 1 → more exploration: rare moves get a boost—useful for self-play or when you want variety.
            # 0 < 𝜏 < 1 → more exploitation: accentuate the top move, but keep some randomness.
            probabilities = visits ** (1.0 / temperature)    # Exponentiation
            probabilities /= probabilities.sum()             # Normalization
            best = np.random.choice(len(actions), p=probabilities)  # Sampling

        # Get the canonical move from the action index
        canonical_move = index_to_move(actions[best], canon.turn)

        # Convert back to the original board perspective
        chosen_move = uncanonicalize_move(canonical_move, self.color)

        return chosen_move

        # chosen_move = index_to_move(actions[best])
        # return mirror_move(chosen_move) if self.color == chess.BLACK else chosen_move

    def _simulate(self, root, board, game_history):
        node = root
        path = []  # list of (node, action)

        # Selection
        while node.is_expanded and list(node.children):
            action, node = self._select_child(node)
            move = index_to_move(action, board.turn)
            board.push(move)
            path.append((node, action))

        #  Evaluation & Expansion
        if board.is_game_over():
            v = self._terminal_value(board)
        else:
            leaf_history = game_history.copy()
            leaf_history.popleft()
            canon_leaf = canonicalize_board(board, self.color)
            leaf_history.append(canon_leaf)
            # leaf_history.append(board.copy())
            # print('leaf', leaf_history)
            v, policy = self._evaluate(board, leaf_history)
            node.P = policy

            self._expand(node, board)

        # Backpropagation
        self._backpropagate(path, v)

    def _select_child(self, node):
        total_N = sum(child.N for child in node.children.values())
        best_score = -float('inf')
        best_action = None
        best_child = None
        for action, child in node.children.items():
            p = node.P[action]
            u = (
                self.c_puct
                * p
                * math.sqrt(total_N)
                / (1 + child.N)
            )
            score = child.Q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def _expand(self, node: Node, board: chess.Board):
        """Generate children for node using board position."""

        for move in board.legal_moves:
            action = move_to_index(move, board)
            board.push(move)
            key = zobrist_hash(board)
            board.pop()

            existing = self.tt.get(key)
            if existing:
                child = existing
                child.parent = node
            else:
                child = Node(key, parent=node)
                self.tt.put(child)
            node.children[action] = child

        node.is_expanded = True

    def _evaluate(self, board: chess.Board, history):
        """Cache-aware CNN evaluation."""
        key = zobrist_hash(board)

        if key in self.cache:
            return self.cache[key]

        input = board_to_tensor(list(history))

        policy, v = self.model.evaluate(input)
        # v, policy = evaluate(input)

        v = max(-1.0, min(1.0, v))

        self.cache[key] = (v, policy)

        return v, policy

    def _add_dirichlet_noise(self, policy: np.ndarray) -> np.ndarray:
        """Mix root priors with Dirichlet noise for exploration."""

        epsilon, alpha = 0.25, 0.03
        noise = np.random.dirichlet([alpha] * len(policy))
        return (1 - epsilon) * policy + epsilon * noise

    def _backpropagate(self, path, value):
        """Propagate evaluation up the tree."""
        for node, _ in reversed(path):
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            value = -value

    def _terminal_value(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -1.0
        # Stalemate or draw
        return 0.0

    def _create_canonical_history(self, board: chess.Board):
        # This is a placeholder - implement based on your create_board_history function
        # Make sure to canonicalize each board state in the history
        history = deque([None]*5, maxlen=5)  # Adjust maxlen as needed

        # Example implementation:
        current_board = board.copy()
        for _ in range(5):  # Or however many history states you need
            canonical_board = canonicalize_board(current_board, self.color)
            history.append(canonical_board)
            if len(current_board.move_stack) == 0:
                break
            current_board.pop()

        return history

def main():
    board = chess.Board()
    cnn = CNNInference(CNN(num_blocks=2), 'cpu', 'chess_model.pth')
    cnn.init_model()
    ai = ChessAI(cnn, color=chess.WHITE, sims=2, c_puct=1.5)
    # ai.select_move(board, use_dirichlet=False, temperature=0.0)

    i = 0
    while not board.is_game_over() and i < 200:
        print(board, i)
        i+=1
        if board.turn == ai.color:
            move = ai.select_move(board, use_dirichlet=False, temperature=0.0)
            print(f"AI plays: {move}")
        else:
            # For demo: random opponent
            move = random.choice(list(board.legal_moves))
            print(f"Opponent plays: {move}")
        board.push(move)

    print("Game over.", board.result())

if __name__ == "__main__":
    main()
