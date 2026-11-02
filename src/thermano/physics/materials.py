"""Material models: k(T, x) and c_p(T) in SI, T in Kelvin.

All models expose the same two methods, so solver and loss code are written
once against this interface and materials are swapped from the config file.
"""

from __future__ import annotations

from dataclasses import dataclass

from thermano.physics._backend import xp
from thermano.registry import MATERIALS


@MATERIALS.register("constant")
@dataclass(frozen=True)
class ConstantMaterial:
    density: float
    conductivity_value: float = 15.0
    heat_capacity_value: float = 500.0

    def conductivity(self, T, x=None):
        return self.conductivity_value + 0.0 * T

    def heat_capacity(self, T):
        return self.heat_capacity_value + 0.0 * T


@MATERIALS.register("hastelloy_x")
@dataclass(frozen=True)
class HastelloyX:
    """Temperature-dependent Hastelloy X (paper Eq. 21; latent heat in apparent c_p)."""

    density: float = 8351.91

    def conductivity(self, T, x=None):
        m = xp(T)
        return 229.87 + 0.0184 * T + 225.10 * m.tanh(0.0118 * (T - 1816.8))

    def heat_capacity(self, T):
        m = xp(T)
        return (
            407.62
            + 0.142 * T
            - 61.43 * m.exp(-3.1e-4 * (T - 798.0) ** 2)
            + 1054.96 * m.exp(-6.2e-5 * (T - 1816.8) ** 2)
        )


@MATERIALS.register("graded_conductivity")
@dataclass(frozen=True)
class GradedConductivity:
    """Two-material layout: conductivity k_in inside x in [x1, x2], k_out outside.

    Smooth tanh blend (sharpness ``beta``) keeps the residual differentiable.
    c_p follows Hastelloy X.
    """

    density: float = 8351.91
    k_out: float = 25.0
    k_in: float = 7.5
    x1: float = 0.8
    x2: float = 1.2
    beta: float = 30.0

    def conductivity(self, T, x=None):
        m = xp(T, x)
        blend = 0.5 * (m.tanh(self.beta * (x - self.x1)) + m.tanh(self.beta * (self.x2 - x)))
        return self.k_out + (self.k_in - self.k_out) * blend + 0.0 * T

    def heat_capacity(self, T):
        return HastelloyX(self.density).heat_capacity(T)


def build_material(cfg):
    return MATERIALS.build(cfg.type, density=cfg.density, **cfg.params)
