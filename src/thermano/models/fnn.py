"""Fully connected network used for branch, trunk and root sub-networks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class FNN(nn.Module):
    """Linear -> activation stack with a linear output layer.

    ``widths = [in, h1, ..., hk, out]``. The activation module is shared across
    layers, so a trainable activation (e.g. ``Sin``) has one parameter per net.
    """

    def __init__(self, widths: Sequence[int], activation: nn.Module) -> None:
        super().__init__()
        if len(widths) < 2:
            raise ValueError("FNN needs at least input and output width")
        self.hidden = nn.ModuleList(
            nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 2)
        )
        self.out = nn.Linear(widths[-2], widths[-1])
        self.activation = activation
        for layer in [*self.hidden, self.out]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden:
            x = self.activation(layer(x))
        return self.out(x)
