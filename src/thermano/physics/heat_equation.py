"""Residual of the scaled transient heat equation (paper Eqs. 2 and 11).

With T = T_s * u and x = L_s * x_hat, the strong form

    rho c_p dT/dt = div(k grad T) + q_v

becomes, after division by rho c_p T_s,

    du/dt - div_hat(k grad_hat u) / (rho c_p L_s^2) - q_v / (rho c_p T_s) = 0 .

Derivatives are taken with automatic differentiation w.r.t. the trunk input.
"""

from __future__ import annotations

import torch

from thermano.physics.heat_source import gaussian_heat_source


def _grad(out: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(out, inp, torch.ones_like(out), create_graph=True)[0]


def heat_equation_residual(u, coords, laser_state, material, laser_cfg, domain_cfg):
    """Pointwise residual.

    u           : (N, 1) network output, scaled temperature
    coords      : (N, 4) trunk input (t, x, y, z) with ``requires_grad=True``
    laser_state : (N, 3) laser centre (x_l, y_l) and absorbed power at time t
    """
    T = u[:, 0] * domain_cfg.temperature_scale
    x = coords[:, 1]
    du = _grad(u, coords)  # (N, 4)
    k = material.conductivity(T, x)
    flux = k.unsqueeze(1) * du[:, 1:4]
    div = sum(_grad(flux[:, i], coords)[:, i + 1] for i in range(3))
    rho_cp = material.density * material.heat_capacity(T)
    q = gaussian_heat_source(
        x, coords[:, 2], coords[:, 3],
        laser_state[:, 0], laser_state[:, 1], laser_state[:, 2],
        radius=laser_cfg.radius, depth=laser_cfg.penetration_depth,
        top=domain_cfg.height, length_scale=domain_cfg.length_scale,
    )
    return (
        du[:, 0]
        - div / (rho_cp * domain_cfg.length_scale**2)
        - q / (rho_cp * domain_cfg.temperature_scale)
    )
