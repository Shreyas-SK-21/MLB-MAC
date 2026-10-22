"""
resnet20.py
===========
Proper ResNet-20 for CIFAR-10 following He et al. (2016) / Idelbayev (2018).

Total parameters: ~272,474.
Supports 3 precision configurations:
  1. mode="fp32": Full FP32 floating point baseline.
  2. mode="fp8": Native FP8 baseline (quantized inputs & weights, standard matmul).
  3. mode="bfp_mlb": Proposed BFP-aligned Multi-Level Binary (MLB) core with block size N
                     and optional offline weight-exponent co-sorting (sort_by_exp).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import (
    MLBConv2d, MLBLinear,
    FP8OnlyConv2d, FP8OnlyLinear,
)


class LambdaLayer(nn.Module):
    """Option A shortcut for CIFAR-10 ResNet: identity with zero-padding."""
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride=1, mode="bfp_mlb", N=64, sort_by_exp=True):
        super().__init__()
        self.stride = stride
        self.mode = mode

        if mode == "bfp_mlb":
            self.conv1 = MLBConv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False, N=N, sort_by_exp=sort_by_exp)
            self.conv2 = MLBConv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, N=N, sort_by_exp=sort_by_exp)
        elif mode == "fp8":
            self.conv1 = FP8OnlyConv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
            self.conv2 = FP8OnlyConv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        else: # fp32
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Option A shortcut matching canonical CIFAR-10 ResNet (Idelbayev)
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, (out_channels - in_channels) // 2, (out_channels - in_channels) // 2), "constant", 0)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20(nn.Module):
    """
    ResNet-20 for CIFAR-10.
    Layers:
      - conv1: 3 -> 16
      - layer1: 3 basic blocks (16 -> 16)
      - layer2: 3 basic blocks (16 -> 32)
      - layer3: 3 basic blocks (32 -> 64)
      - fc: 64 -> 10
    """
    def __init__(self, num_classes=10, mode="bfp_mlb", N=64, sort_by_exp=True):
        super().__init__()
        self.mode = mode
        self.N = N
        self.sort_by_exp = sort_by_exp

        if mode == "bfp_mlb":
            self.conv1 = MLBConv2d(3, 16, 3, stride=1, padding=1, bias=False, N=N, sort_by_exp=sort_by_exp)
        elif mode == "fp8":
            self.conv1 = FP8OnlyConv2d(3, 16, 3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 16, 3, stride=1, mode=mode, N=N, sort_by_exp=sort_by_exp)
        self.layer2 = self._make_layer(16, 32, 3, stride=2, mode=mode, N=N, sort_by_exp=sort_by_exp)
        self.layer3 = self._make_layer(32, 64, 3, stride=2, mode=mode, N=N, sort_by_exp=sort_by_exp)

        if mode == "bfp_mlb":
            self.linear = MLBLinear(64, num_classes, bias=True, N=N, sort_by_exp=sort_by_exp)
        elif mode == "fp8":
            self.linear = FP8OnlyLinear(64, num_classes, bias=True)
        else:
            self.linear = nn.Linear(64, num_classes)

    @staticmethod
    def _make_layer(in_channels, out_channels, num_blocks, stride, mode, N, sort_by_exp):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        ch_in = in_channels
        for s in strides:
            layers.append(BasicBlock(ch_in, out_channels, s, mode=mode, N=N, sort_by_exp=sort_by_exp))
            ch_in = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_resnet20(mode="bfp_mlb", N=64, sort_by_exp=True, num_classes=10):
    return ResNet20(num_classes=num_classes, mode=mode, N=N, sort_by_exp=sort_by_exp)
