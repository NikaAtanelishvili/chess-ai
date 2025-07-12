import time

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
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion_policy = nn.NLLLoss()
        self.criterion_value = nn.MSELoss()
        self.history = {'policy_loss': [], 'value_loss': []}

    def train(self, num_epochs: int):
        for epoch in range(1, num_epochs + 1):
            avg_pl, avg_vl = self.train_epoch(epoch, num_epochs)
            # final summary line for this epoch
            print(f"Epoch {epoch}/{num_epochs}  "
                  f"— avg policy loss: {avg_pl:.4f}, "
                  f"avg value loss: {avg_vl:.4f}")

    def train_epoch(self, epoch: int, num_epochs: int):
        self.model.train()
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        samples_done      = 0

        for batch_idx, (x_batch, pi_batch, v_batch) in enumerate(self.loader, start=1):
            x_batch    = x_batch.to(self.device)
            target_idx = pi_batch.argmax(dim=1).to(self.device)
            v_batch    = v_batch.to(self.device)

            self.optimizer.zero_grad()
            log_p, v_pred       = self.model(x_batch)
            policy_loss         = self.criterion_policy(log_p, target_idx)
            value_loss          = self.criterion_value(v_pred, v_batch)
            (policy_loss + value_loss).backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss  += value_loss.item()

            # update sample count and print progress
            samples_done += x_batch.size(0)
            print(
                f"\rEpoch {epoch}/{num_epochs}  "
                f"Samples {samples_done}/{self.num_samples}",
                end=""
            )

        # finish the line
        print()

        avg_pl = total_policy_loss / len(self.loader)
        avg_vl = total_value_loss  / len(self.loader)
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
    pgn_paths = ['lichess_test.pgn']
    dataset = LichessPGNDataset(pgn_paths)
    print(len(dataset))
    print(dataset[0])
    # loader = DataLoader(dataset, batch_size=1, shuffle=True)
    # print(next(loader))
    return

    model = CNN(num_blocks=2)
    x, policy_target, value_target = dataset[0]
    x = x.unsqueeze(0)  # Add batch dimension -> (1,26,8,8)
    model.eval()
    with torch.no_grad():
        start = time.time()
        log_probs, value = model(x)  # log_probs: (1,4672), value: (1,)
        end = time.time()
    # if you just want the 1D outputs:
    log_probs = log_probs.squeeze(0)  # -> (4672,)
    value = value.item()  # -> float

    print(f'{end - start:.6f}')


    print(log_probs, value)
    # trainer = Trainer(model, dataset)

    # epochs = 1
    # for epoch in range(1, epochs+1):
    #     pl, vl = trainer.train_epoch()
    #     print(f"Epoch {epoch}: Policy Loss={pl:.4f}, Value Loss={vl:.4f}")
    #
    # trainer.save('chess_model.pth')
    # plot_losses(trainer.history, save_path='loss_plot.png')

if __name__ == '__main__':
    main()
