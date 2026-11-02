"""Activation functions with a trainable frequency/slope parameter."""

from __future__ import annotations

import math

import torch
from torch import nn

from thermano.registry import ACTIVATIONS


@ACTIVATIONS.register("sin")
class Sin(nn.Module):
    """sin(pi * beta * x) with trainable beta (smooth, so second derivatives exist)."""

    def __init__(self, beta_init: float = 1.0) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(math.pi * self.beta * x)


@ACTIVATIONS.register("swish")
class Swish(nn.Module):
    def __init__(self, beta_init: float = 1.0) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


@ACTIVATIONS.register("tanh")
def _tanh() -> nn.Module:
    return nn.Tanh()
