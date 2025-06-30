import torch
from torch import nn
import torch.nn.functional as F
from src.mcts2 import POLICY_SIZE


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class CNN(nn.Module):
    def __init__(self, in_planes=26, num_blocks=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(num_blocks)])
        # Policy head
        self.policy_conv = nn.Conv2d(256, 2*73, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2*73)
        self.policy_fc = nn.Linear(2*73*8*8, POLICY_SIZE)
        # Value head
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1*8*8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # x: (batch,26,8,8)
        x = self.stem(x)
        x = self.res_blocks(x)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))  # (batch,146,8,8)
        p = p.view(p.size(0), -1)                        # flatten to (batch,146*8*8)
        p = self.policy_fc(p)                            # (batch,4672)
        p = F.log_softmax(p, dim=1)

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))    # (batch,1,8,8)
        v = v.view(v.size(0), -1)                        # (batch,64)
        v = F.relu(self.value_fc1(v))                    # (batch,256)
        v = torch.tanh(self.value_fc2(v)).view(-1)       # (batch,)

        return p, v