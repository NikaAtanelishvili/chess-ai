import chess
import numpy as np
import math
import random

class Node:
    def __init__(self, fen, parent=None):
        self.fen = fen               # Board position FEN
        self.parent = parent         # Parent node
        self.children = {}           # move_uci -> Node
        self.P = {}                  # Prior policy P(s,a) | move_uci -> (P, v)
        self.N = 0                   # Visit count
        self.W = 0.0                 # Total value
        self.Q = 0.0                 # Mean value W/N
        self.is_expanded = False

class TranspositionTable:
    def __init__(self):
        self.table = {}             # fen -> Node

    def get(self, fen):
        return self.table.get(fen)

    def put(self, node):
        self.table[node.fen] = node

class CNN:
    def __init__(self, model_path=None):
        # TODO: load your trained network here (PyTorch, TensorFlow, etc.)
        pass

    def evaluate(self, board):
        """
        Stub evaluation. Replace with real CNN inference:
        - Returns value v in [-1,1]
        - Returns policy dict mapping move_uci to probability
        """
        legal = list(board.legal_moves)
        if not legal:
            return 0.0, {}
        # Uniform policy stub
        policy = {m.uci(): 1.0 / len(legal) for m in legal}
        # Random value stub
        v = random.uniform(-1, 1)
        return v, policy

class ChessAI:
    def __init__(self, model, sims=800, c_puct=1.0):
        self.model = model
        self.sims = sims
        self.c_puct = c_puct
        self.tt = TranspositionTable()
        self.cache = {}             # fen -> (v, policy)

    def select_move(self, board):
        root = Node(board.fen())
        self.tt.put(root)

        # Initial evaluation & expansion of root
        v_root, policy_root = self._evaluate(board)

        root.P = self._add_dirichlet_noise(policy_root)

        self._expand(root, board)

        # MCTS simulations
        for _ in range(self.sims):
            board_copy = board.copy()
            self._simulate(root, board_copy)

        # Pick most visited move
        best_move_uci, best_child = max(
            root.children.items(), key=lambda kv: kv[1].N
        )

        # Tree reuse: detach selected child
        best_child.parent = None
        return chess.Move.from_uci(best_move_uci)

    def _simulate(self, root, board):
        node = root
        path = [node]

        # Selection
        while node.is_expanded and list(node.children):
            move_uci, node = self._select_child(node)
            board.push_uci(move_uci)
            path.append(node)

        # Evaluation & Expansion
        if board.is_game_over():
            v = self._terminal_value(board)
        else:
            v, policy = self._evaluate(board)
            node.P = policy
            self._expand(node, board)

        # Backpropagation
        self._backpropagate(path, v)

    def _select_child(self, node):
        total_N = sum(child.N for child in node.children.values())
        best_score = -float('inf')
        best_move = None
        best_child = None
        for move_uci, child in node.children.items():
            u = (
                self.c_puct
                * node.P.get(move_uci, 0)
                * math.sqrt(total_N)
                / (1 + child.N)
            )
            score = child.Q + u
            if score > best_score:
                best_score = score
                best_move = move_uci
                best_child = child
        return best_move, best_child

    def _expand(self, node, board):
        """Generate children for node using board position."""
        for move in board.legal_moves:
            m = move.uci()
            if m not in node.children:
                board.push(move)
                fen2 = board.fen()
                board.pop()

                existing = self.tt.get(fen2)
                if existing:
                    child = existing
                    child.parent = node
                else:
                    child = Node(fen2, parent=node)
                    self.tt.put(child)

                node.children[m] = child
        node.is_expanded = True

    def _evaluate(self, board):
        """Cache-aware CNN evaluation."""
        fen = board.fen()

        if fen in self.cache:
            return self.cache[fen]

        v, policy = self.model.evaluate(board)

        # Normalize policy to sum=1
        total = sum(policy.values()) or 1.0

        policy = {m: p / total for m, p in policy.items()}

        # Clip value to [-1,1]
        v = max(-1.0, min(1.0, v))

        self.cache[fen] = (v, policy)

        return v, policy

    def _add_dirichlet_noise(self, policy):
        """Mix root priors with Dirichlet noise for exploration."""
        epsilon = 0.25
        alpha = 0.03
        moves = list(policy.keys())
        noise = np.random.dirichlet([alpha] * len(moves))
        return {
            m: (1 - epsilon) * policy[m] + epsilon * noise[i]
            for i, m in enumerate(moves)
        }

    def _backpropagate(self, path, value):
        for node in reversed(path):
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            value = -value

    def _terminal_value(self, board):
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
            move = ai.select_move(board)
            print(f"AI plays: {move}")
        else:
            # For demo: random opponent
            move = random.choice(list(board.legal_moves))
            print(f"Opponent plays: {move}")
        board.push(move)

    print("Game over.", board.result())

if __name__ == "__main__":
    main()
