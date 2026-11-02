"""Loss functions.

``PhysicsInformedLoss`` implements paper Eqs. 7-14: initial-condition,
Dirichlet / Neumann boundary and PDE-residual terms, weighted by the inverse of
their values at the start of training. ``DataLoss`` is the supervised MSE used
in the data-driven mode. Both return ``(total, {term: value})`` so the trainer
can log every term without knowing which loss it is running.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from thermano.physics.heat_equation import heat_equation_residual


@dataclass
class PointSet:
    branch: torch.Tensor
    coords: torch.Tensor
    extra: torch.Tensor | None = None  # laser state / IC target / normals


class PhysicsInformedLoss:
    def __init__(self, interior, initial, neumann, dirichlet, material, laser_cfg, domain_cfg,
                 weighting: str = "initial_loss") -> None:
        self.interior, self.initial = interior, initial
        self.neumann, self.dirichlet = neumann, dirichlet
        self.material, self.laser_cfg, self.domain_cfg = material, laser_cfg, domain_cfg
        self.weighting = weighting
        self.weights: dict[str, float] | None = None
        self.T0 = domain_cfg.initial_temperature / domain_cfg.temperature_scale

    def terms(self, model) -> dict[str, torch.Tensor]:
        p = self.interior
        u = model(p.branch, p.coords)
        res = heat_equation_residual(u, p.coords, p.extra, self.material, self.laser_cfg, self.domain_cfg)
        out = {"pde": (res**2).mean()}

        ic = self.initial
        out["ic"] = ((model(ic.branch, ic.coords) - ic.extra) ** 2).mean()

        bc = []
        if self.neumann is not None:
            n = self.neumann
            un = model(n.branch, n.coords)
            g = torch.autograd.grad(un, n.coords, torch.ones_like(un), create_graph=True)[0]
            bc.append(((g * n.extra).sum(dim=1) ** 2).mean())
        if self.dirichlet is not None:
            d = self.dirichlet
            bc.append(((model(d.branch, d.coords) - self.T0) ** 2).mean())
        out["bc"] = sum(bc) if bc else torch.zeros((), device=u.device)
        return out

    def __call__(self, model):
        t = self.terms(model)
        if self.weights is None:
            if self.weighting == "initial_loss":  # paper Eq. 14
                self.weights = {k: 1.0 / max(float(v.detach()), 1e-12) for k, v in t.items()}
            else:
                self.weights = {k: 1.0 for k in t}
        total = sum(self.weights[k] * v for k, v in t.items())
        return total, {k: float(v.detach()) for k, v in t.items()}


class DataLoss:
    def __init__(self, branch, coords, target) -> None:
        self.branch, self.coords, self.target = branch, coords, target

    def __call__(self, model):
        mse = ((model(self.branch, self.coords) - self.target) ** 2).mean()
        return mse, {"data": float(mse.detach())}
