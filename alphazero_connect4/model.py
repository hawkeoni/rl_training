import torch
import torch.nn as nn
import torch.nn.functional as F

from .game import ROWS, COLS


class ResBlock(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(channels)
        self.conv2: nn.Conv2d = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class AlphaZeroNet(nn.Module):

    def __init__(self, num_res_blocks: int = 3, channels: int = 128) -> None:
        super().__init__()
        self.conv_in: nn.Conv2d = nn.Conv2d(2, channels, 3, padding=1, bias=False)
        self.bn_in: nn.BatchNorm2d = nn.BatchNorm2d(channels)
        self.res_blocks: nn.Sequential = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])
        # Policy head
        self.policy_conv: nn.Conv2d = nn.Conv2d(channels, 32, 1, bias=False)
        self.policy_bn: nn.BatchNorm2d = nn.BatchNorm2d(32)
        self.policy_fc: nn.Linear = nn.Linear(32 * ROWS * COLS, COLS)
        # Value head
        self.value_conv: nn.Conv2d = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn: nn.BatchNorm2d = nn.BatchNorm2d(1)
        self.value_fc1: nn.Linear = nn.Linear(ROWS * COLS, 64)
        self.value_fc2: nn.Linear = nn.Linear(64, 1)

    def forward_policy(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        return self.policy_fc(p)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        return p, v
