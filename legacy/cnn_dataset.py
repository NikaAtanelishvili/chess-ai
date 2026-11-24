import re
from collections import deque

import torch
from torch.utils.data import Dataset

from src.mcts2 import move_to_index, POLICY_SIZE
from src.utils import iter_games, board_to_tensor


class LichessPGNDataset(Dataset):
    def __init__(self, pgn_paths, history_size=5, max_eval=10.0, transform=None):
        """
        pgn_paths: list of file paths to PGN files
        history_size: number of positions to include (current + last history_size-1)
        max_eval: clamp engine evals (in pawns) to [-max_eval, max_eval] then normalize to [-1,1]
        transform: optional transform on feature tensor
        """
        self.samples = []  # list of (history_boards, action_idx, value_norm) # 4d tensor, int, int
        self.transform = transform
        self.history_size = history_size
        self.max_eval = max_eval

        eval_pattern = re.compile(r"\[?%eval\s+([-+]?\d+\.?\d*)]?")

        for game in iter_games(pgn_paths):
            board = game.board()
            history = deque([None] * history_size, maxlen=history_size)

            for node in game.mainline():
                move = node.move
                comment = node.comment or ''
                match = eval_pattern.search(comment)
                board.push(move)
                history.append(board.copy())

                if not match:
                    continue

                # Normalize evaluation to [-1,1]
                eval_score = float(match.group(1))
                eval_score = max(-self.max_eval, min(self.max_eval, eval_score))
                value_norm = eval_score / self.max_eval

                # Policy label via move_to_index
                try:
                    action_idx = move_to_index(move)
                except ValueError:
                    continue
                self.samples.append((list(history), action_idx, value_norm))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        history_boards, action_idx, value_norm = self.samples[idx]

        # Convert history to feature tensor
        x = board_to_tensor(history_boards)
        if self.transform:
            x = self.transform(x)

        # Policy target: one-hot
        policy_target = torch.zeros(POLICY_SIZE, dtype=torch.float32)
        policy_target[action_idx] = 1.0

        # Value target: scalar
        value_target = torch.tensor(value_norm, dtype=torch.float32)
        return x, policy_target, value_target
