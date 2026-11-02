"""DeepONet and enriched DeepONet (En-DeepONet) operator networks.

Both map an input *function* sampled at fixed sensors (branch input: laser path
or laser-power profile) and a query point ``(t, x, y, z)`` (trunk input) to the
scaled temperature at that point.

* ``DeepONet``  : T = sum_k B_k * T_k                       (Lu et al., 2021)
* ``EnDeepONet``: T = w_x.(B*T) + w_+.(B+T) + w_-.(B-T)     (Haghighat et al., 2024)

Input normalisation is part of the model (registered buffers), so a saved
``state_dict`` is self-contained and cannot be paired with the wrong scaling.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from thermano.models.fnn import FNN
from thermano.registry import ACTIVATIONS, MODELS


class InputScaler(nn.Module):
    """Affine map of ``[low, high]`` onto ``[-1, 1]``, stored as buffers."""

    def __init__(self, low: Sequence[float], high: Sequence[float]) -> None:
        super().__init__()
        low_t = torch.as_tensor(low, dtype=torch.get_default_dtype())
        high_t = torch.as_tensor(high, dtype=torch.get_default_dtype())
        span = torch.where(high_t > low_t, high_t - low_t, torch.ones_like(high_t))
        self.register_buffer("scale", 2.0 / span)
        self.register_buffer("bias", -2.0 * low_t / span - 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.bias


class _OperatorBase(nn.Module):
    def __init__(
        self,
        n_sensors: int,
        n_coords: int,
        branch_layers: Sequence[int],
        trunk_layers: Sequence[int],
        latent_dim: int,
        activation: str,
        branch_bounds: tuple[Sequence[float], Sequence[float]],
        trunk_bounds: tuple[Sequence[float], Sequence[float]],
        output_transform: str = "softplus",
    ) -> None:
        super().__init__()
        if output_transform not in {"softplus", "identity"}:
            raise ValueError(f"unknown output_transform '{output_transform}'")
        self.branch_scaler = InputScaler(*branch_bounds)
        self.trunk_scaler = InputScaler(*trunk_bounds)
        self.branch = FNN([n_sensors, *branch_layers, latent_dim], ACTIVATIONS.build(activation))
        self.trunk = FNN([n_coords, *trunk_layers, latent_dim], ACTIVATIONS.build(activation))
        self.output_transform = output_transform

    def _latent(self, u: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.branch(self.branch_scaler(u)), self.trunk(self.trunk_scaler(y))

    def _finish(self, out: torch.Tensor) -> torch.Tensor:
        # softplus keeps the (scaled, above-ambient) temperature non-negative
        return F.softplus(out) if self.output_transform == "softplus" else out


@MODELS.register("deeponet")
class DeepONet(_OperatorBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b, t = self._latent(u, y)
        return self._finish((b * t).sum(dim=-1, keepdim=True) + self.bias)


@MODELS.register("en_deeponet")
class EnDeepONet(_OperatorBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        p = kwargs["latent_dim"]
        # "root net": one linear read-out per enrichment operator
        self.root_mul = nn.Linear(p, 1)
        self.root_add = nn.Linear(p, 1)
        self.root_sub = nn.Linear(p, 1)

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b, t = self._latent(u, y)
        out = self.root_mul(b * t) + self.root_add(b + t) + self.root_sub(b - t)
        return self._finish(out)


def build_model(
    model_cfg,
    n_sensors: int,
    branch_bounds: tuple[Sequence[float], Sequence[float]],
    trunk_bounds: tuple[Sequence[float], Sequence[float]],
    n_coords: int = 4,
) -> _OperatorBase:
    return MODELS.build(
        model_cfg.type,
        n_sensors=n_sensors,
        n_coords=n_coords,
        branch_layers=model_cfg.branch_layers,
        trunk_layers=model_cfg.trunk_layers,
        latent_dim=model_cfg.latent_dim,
        activation=model_cfg.activation,
        branch_bounds=branch_bounds,
        trunk_bounds=trunk_bounds,
        output_transform=model_cfg.output_transform,
    )


@MODELS.register("pinn")
class PINN(_OperatorBase):
    """Plain coordinate network (no branch): one solution per scenario / track.

    Shares the operator interface ``forward(u, y)`` so that the trainer, losses
    and evaluation code are identical for PINN and DeepONet; ``u`` is ignored.
    """

    def __init__(self, **kwargs) -> None:
        kwargs["n_sensors"] = 1
        kwargs["branch_bounds"] = ([0.0], [1.0])
        super().__init__(**kwargs)
        del self.branch, self.branch_scaler
        self.head = torch.nn.Linear(kwargs["latent_dim"], 1)

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._finish(self.head(self.trunk(self.trunk_scaler(y))))
