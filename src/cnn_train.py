import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.cnn import CNN
from src.cnn_dataset import LichessPGNDataset
from src.mcts2 import move_to_index, index_to_move


class Trainer:
    def __init__(self, model, dataset, batch_size=1, lr=1e-3, device=None):
        # self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cuda'
        self.model = model.to(self.device)
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion_policy = nn.NLLLoss()
        self.criterion_value = nn.MSELoss()
        self.history = {'policy_loss': [], 'value_loss': []}

    def train_epoch(self):
        self.model.train()
        total_policy_loss = 0
        total_value_loss = 0
        for x_batch, pi_batch, v_batch in self.loader:
            x_batch = x_batch.to(self.device)
            # Convert one-hot policy to index
            target_idx = pi_batch.argmax(dim=1).to(self.device)
            v_batch = v_batch.to(self.device)

            self.optimizer.zero_grad()
            log_p, v_pred = self.model(x_batch)
            policy_loss = self.criterion_policy(log_p, target_idx)
            value_loss = self.criterion_value(v_pred, v_batch)
            (policy_loss + value_loss).backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        avg_pl = total_policy_loss / len(self.loader)
        avg_vl = total_value_loss / len(self.loader)
        self.history['policy_loss'].append(avg_pl)
        self.history['value_loss'].append(avg_vl)
        return avg_pl, avg_vl

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)

# Visualization utilities

def plot_losses(history, save_path=None):
    epochs = range(1, len(history['policy_loss'])+1)
    plt.figure()
    plt.plot(epochs, history['policy_loss'], label='Policy Loss')
    plt.plot(epochs, history['value_loss'], label='Value Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def mask_illegal_moves(board, log_p):
    mask = torch.full_like(log_p, float('-inf'))
    for move in board.legal_moves:
        idx = move_to_index(move)
        mask[idx] = log_p[idx]
    return torch.softmax(mask, dim=0)


def visualize_top_n(board, probs, n=5):
    top_idxs = torch.topk(probs, n).indices.tolist()
    for idx in top_idxs:
        move = index_to_move(idx)
        print(f"{move}: {probs[idx]:.4f}")


def saliency_map(model, x, action_idx):
    x = x.unsqueeze(0).clone().detach().requires_grad_(True)
    log_p, _ = model(x)
    score = log_p[0, action_idx]
    score.backward()
    saliency = x.grad.abs().sum(dim=1)[0]
    plt.imshow(saliency.cpu(), cmap='hot')
    plt.colorbar()
    plt.show()

# Example training script
def main():
    pgn_paths = ['lichess_db_standard_rated_eval_test_2013-01.pgn']
    dataset = LichessPGNDataset(pgn_paths)
    model = CNN(num_blocks=2)
    trainer = Trainer(model, dataset)

    epochs = 1
    for epoch in range(1, epochs+1):
        pl, vl = trainer.train_epoch()
        print(f"Epoch {epoch}: Policy Loss={pl:.4f}, Value Loss={vl:.4f}")

    trainer.save('chess_model.pth')
    plot_losses(trainer.history, save_path='loss_plot.png')

if __name__ == '__main__':
    main()
