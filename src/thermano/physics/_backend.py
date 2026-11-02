"""Let physics kernels run unchanged on NumPy arrays (FD solver) and torch tensors (PINN loss)."""

from __future__ import annotations

import numpy as np
import torch


def xp(*arrays):
    return torch if any(isinstance(a, torch.Tensor) for a in arrays) else np
