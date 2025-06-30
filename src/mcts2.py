#!/usr/bin/env python3
import chess
import numpy as np
import math
import random

from chess.polyglot import zobrist_hash

def mirror_move(move: chess.Move) -> chess.Move:
    from_sq = chess.square_mirror(move.from_square)
    to_sq = chess.square_mirror(move.to_square)
    return chess.Move(from_sq, to_sq, promotion=move.promotion)

# Action encoding: 8×8×73 directional move representation
DELTAS = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
KNIGHT_DELTAS = [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]
ACTIONS = []

# Sliding moves: 8 directions × distances 1–7
for dx, dy in DELTAS:
    for dist in range(1, 8):
        ACTIONS.append((dx * dist, dy * dist, False, None))

# Knight jumps: 8
for dx, dy in KNIGHT_DELTAS:
    ACTIONS.append((dx, dy, False, None))

UNDERPROM_DIRS = [(0,1),(1,1),(-1,1)]  # relative to white; board is canonicalized
for promo_piece in (chess.ROOK, chess.BISHOP, chess.KNIGHT):
    for dx, dy in UNDERPROM_DIRS:
        ACTIONS.append((dx, dy, True, promo_piece))

assert len(ACTIONS) == 73, f"Expected 73 actions, got {len(ACTIONS)}"

POLICY_SIZE = 64 * len(ACTIONS)  # 4672

# Move-index conversions
def move_to_index(move: chess.Move) -> int:
    from_sq = move.from_square
    to_sq = move.to_square

    fr_r, fr_c = divmod(from_sq, 8)
    to_r, to_c = divmod(to_sq, 8)

    dx, dy = to_r - fr_r, to_c - fr_c

    is_promo = move.promotion is not None

    for aid, (adx, ady, promo_flag, promo_piece) in enumerate(ACTIONS):
        if (adx, ady, promo_flag) == (dx, dy, is_promo):
            if promo_flag:
                if move.promotion == promo_piece:
                    return from_sq * len(ACTIONS) + aid
            else:
                return from_sq * len(ACTIONS) + aid

    raise ValueError(f"Illegal move for ACTIONS: {move}")

def index_to_move(index: int) -> chess.Move:
    n = len(ACTIONS)
    from_sq = index // n
    aid = index % n
    dx, dy, promo_flag, promo_piece = ACTIONS[aid]

    fr_r, fr_c = divmod(from_sq, 8)
    to_r, to_c = fr_r + dx, fr_c + dy

    if not (0 <= to_r < 8 and 0 <= to_c < 8):
        return None

    to_sq = to_r * 8 + to_c

    promotion = promo_piece if promo_flag else None

    # any sliding ACTION with promo_flag=False that moves a pawn onto the last rank will by convention be treated
    # as a *queen* promotion. When you push it on a board where a pawn arrives at the last rank,
    # python-chess will default that promotion to a queen.
    return chess.Move(from_sq, to_sq, promotion=promotion)

class Node:
    def __init__(self, key: int, parent=None):
        self.key = key                                      # Zobrist hash of position
        self.parent = parent                                # Parent node
        self.children = {}                                  # action_index -> Node
        self.P = np.zeros(POLICY_SIZE, dtype=np.float32)    # Prior policy P(s,a) | Prior probability vector
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

class CNN:
    def __init__(self, model_path=None):
        # TODO: trained network here (PyTorch)
        pass

    # DUMMY EVALUATION
    def evaluate(self, board: chess.Board):
        """
        Returns:
          v: float in [-1,1]
          policy: np.array of shape (POLICY_SIZE,) summing to 1
        """
        v = random.uniform(-1, 1)
        policy = np.zeros(POLICY_SIZE, dtype=np.float32)

        print(board.turn)
        for move in board.legal_moves:
            idx = move_to_index(move)
            policy[idx] = 1.0

        total = policy.sum() or 1.0
        policy /= total

        return v, policy


class ChessAI:
    def __init__(self, model, color=chess.WHITE, sims=800, c_puct=1.0):
        self.model = model
        self.color = color
        self.sims = sims
        self.c_puct = c_puct
        self.tt = TranspositionTable()
        self.cache = {}             # key:int -> (v, policy)

    def select_move(self, board: chess.Board, use_dirichlet=False, temperature=0.0) -> chess.Move:
        canon = board.mirror() if self.color == chess.BLACK else board
        root_key = zobrist_hash(canon)
        root = self.tt.get(root_key) or Node(root_key)
        root.parent = None

        self.tt.put(root)

        # Initial evaluation
        v_root, policy_root = self._evaluate(canon)

        if use_dirichlet:
            policy_root = self._add_dirichlet_noise(policy_root)

        root.P = policy_root

        # expansion
        self._expand(root, canon)

        # MCTS simulations
        for _ in range(self.sims):
            board_copy = canon.copy()
            self._simulate(root, board_copy)
        
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

        chosen_move = index_to_move(actions[best])

        return mirror_move(chosen_move) if self.color == chess.BLACK else chosen_move

    def _simulate(self, root, board):
        node = root
        path = []  # list of (node, action)

        # --- Selection ---
        while node.is_expanded and list(node.children):
            action, node = self._select_child(node)
            move = index_to_move(action)
            board.push(move)
            path.append((node, action))

        # --- Evaluation & Expansion ---
        if board.is_game_over():
            v = self._terminal_value(board)
        else:
            v, policy = self._evaluate(board)
            node.P = policy
            self._expand(node, board)

        # --- Backpropagation ---
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
            action = move_to_index(move)
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

    def _evaluate(self, board: chess.Board):
        """Cache-aware CNN evaluation."""
        key = zobrist_hash(board)

        if key in self.cache:
            return self.cache[key]

        v, policy = self.model.evaluate(board)

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


def main():
    board = chess.Board()
    ai = ChessAI(CNN(), color=chess.WHITE, sims=10, c_puct=1.5)

    i = 0
    while not board.is_game_over() or i < 10:
        print(board)
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
