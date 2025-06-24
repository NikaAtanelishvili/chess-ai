#!/usr/bin/env python3
import chess
import numpy as np
import math
import random

from chess.polyglot import zobrist_hash

# --- Move encoding utilities -----------------------------------------------
PROMO_MAP = {None: 0, chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}
INV_PROMO_MAP = {v: k for k, v in PROMO_MAP.items()}

def encode_move(move: chess.Move) -> int:
    """Compact integer code: 6 bits from_square, 6 bits to_square, 4 bits promotion."""
    return move.from_square | (move.to_square << 6) | (PROMO_MAP[move.promotion] << 12)

def decode_move(code: int) -> chess.Move:
    """Reconstruct Move from integer code."""
    from_sq = code & 0x3F
    to_sq = (code >> 6) & 0x3F
    promo_code = (code >> 12) & 0xF
    promotion = INV_PROMO_MAP.get(promo_code, None)
    return chess.Move(from_sq, to_sq, promotion=promotion)

# --- Policy head alignment: global move-index map -------------------------
# Build once: map every possible (from, to, promotion) to a unique index
# POLICY_SIZE = total number of unique move codes

def build_move_index_map():
    idx = 0
    move_index = {}
    for from_sq in range(64):
        for to_sq in range(64):
            # normal (non-promotion)
            code = from_sq | (to_sq << 6)
            move_index[code] = idx; idx += 1
            # promotions
            for promo_code in [1, 2, 3, 4]:
                code_p = from_sq | (to_sq << 6) | (promo_code << 12)
                move_index[code_p] = idx; idx += 1
    return move_index, idx

MOVE_INDEX, POLICY_SIZE = build_move_index_map()

class Node:
    def __init__(self, key: int, parent=None):
        self.key = key                                      # Zobrist hash of position
        self.parent = parent                                # Parent node
        self.children = {}                                  # move_code -> Node
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

        for move in board.legal_moves:
            code = encode_move(move)
            idx = MOVE_INDEX.get(code)

            if idx is not None:
                policy[idx] = 1.0

        total = policy.sum() or 1.0
        policy /= total

        return v, policy

class ChessAI:
    def __init__(self, model, sims=800, c_puct=1.0):
        self.model = model
        self.sims = sims
        self.c_puct = c_puct
        self.tt = TranspositionTable()
        self.cache = {}             # key:int -> (v, policy)

    def select_move(self, board: chess.Board, use_dirichlet=False, temperature=0.0) -> chess.Move:
        root_key = zobrist_hash(board)
        root = self.tt.get(root_key) or Node(root_key)
        root.parent = None

        self.tt.put(root)

        # Initial evaluation
        v_root, policy_root = self._evaluate(board)

        if use_dirichlet:
            policy_root = self._add_dirichlet_noise(policy_root)

        root.P = policy_root

        # expansion
        self._expand(root, board)

        # MCTS simulations
        for _ in range(self.sims):
            board_copy = board.copy()
            self._simulate(root, board_copy)
        
        # NOTE: parallelism is guaranteed in this specific case, but only in modern Python versions.
        # As of Python 3.7+, the built-in dict maintains insertion order as an official language feature.
        # So, root.children.keys() and root.children.values() will return items in corresponding, consistent order.
        visits = np.array([node.N for node in root.children.values()], dtype=np.float32)
        codes = list(root.children.keys())

        if temperature == 0.0:
            best = np.argmax(visits)
        else:
            # Stochastic Sampling (𝜏>0)
            # 𝜏=0 → fully greedy: always the max-visit move (best exploitation).
            # 𝜏 = 1 → raw visits: sample in proportion to exact visit counts.
            # 𝜏 > 1 → more exploration: rare moves get a boost—useful for self-play or when you want variety.
            # 0 < 𝜏 < 1 → more exploitation: accentuate the top move, but keep some randomness.
            probabilities = visits ** (1.0 / temperature)    # Exponentiation
            probabilities /= probabilities.sum()             # Normalization
            best = np.random.choice(codes, p=probabilities)  # Sampling

        chosen_code = codes[best]
        return decode_move(chosen_code)

    def _simulate(self, root, board):
        node = root
        path = []  # list of (node, move_code)

        # --- Selection ---
        while node.is_expanded and list(node.children):
            move_code, node = self._select_child(node)
            board.push(decode_move(move_code))
            path.append((node, move_code))

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
        best_move = None
        best_child = None
        for code, child in node.children.items():
            idx = MOVE_INDEX[code]
            u = (
                self.c_puct
                * node.P[idx]
                * math.sqrt(total_N)
                / (1 + child.N)
            )
            score = child.Q + u
            if score > best_score:
                best_score = score
                best_move = code
                best_child = child
        return best_move, best_child

    def _expand(self, node: Node, board: chess.Board):
        """Generate children for node using board position."""
        for move in board.legal_moves:
            code = encode_move(move)
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
            node.children[code] = child

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
    ai = ChessAI(CNN(), sims=800, c_puct=1.5)
    ai_color = chess.WHITE

    while not board.is_game_over():
        print(board)
        if board.turn == ai_color:
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
