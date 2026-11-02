"""Verification of the finite-difference reference solver."""

import math

import numpy as np
import pytest

from thermano.config import DomainConfig, LaserConfig
from thermano.physics.materials import ConstantMaterial, HastelloyX
from thermano.reference.fd_solver import Grid, solve_fd, stable_time_step

ALL_NEUMANN = {f: "neumann" for f in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")}
MAT = ConstantMaterial(density=8000.0, conductivity_value=20.0, heat_capacity_value=500.0)


def _decay_error(nx: int) -> float:
    d = DomainConfig(length=2.0, width=0.5, height=0.5, boundaries=ALL_NEUMANN)
    lz = LaserConfig(power=0.0)
    grid = Grid.uniform(d, (nx, 3, 3))
    alpha = 20.0 / (8000.0 * 500.0) / d.length_scale**2  # mm^2/s
    t_end = 0.2 * d.length**2 / (alpha * math.pi**2)

    def init(X, Y, Z):
        return 300.0 + 100.0 * np.cos(math.pi * X / d.length)

    res = solve_fd(d, lz, MAT, grid, lambda t: (0.0, 0.0), t_end, [t_end], initial=init, safety=0.2)
    exact = 300.0 + 100.0 * math.exp(-alpha * math.pi**2 * t_end / d.length**2) * np.cos(math.pi * grid.x / d.length)
    return float(np.max(np.abs(res.temperature[-1][:, 1, 1] - exact)))


def test_matches_analytical_decay():
    assert _decay_error(41) < 0.05  # K, on a 100 K amplitude


@pytest.mark.slow
def test_second_order_spatial_convergence():
    e1, e2 = _decay_error(41), _decay_error(81)
    order = math.log2(e1 / e2)
    assert 1.8 < order < 2.2, order  # observed 2.11


def test_energy_conservation_insulated_box():
    """All faces insulated: stored energy must equal injected energy (P * c / r * t)."""
    d = DomainConfig(length=2.0, width=1.5, height=1.0, boundaries=ALL_NEUMANN)
    lz = LaserConfig(power=150.0, radius=0.45, penetration_depth=0.45)
    grid = Grid.uniform(d, (41, 31, 21))
    t_end = 2e-3
    res = solve_fd(d, lz, MAT, grid, lambda t: (1.0, 0.75), t_end, [t_end])
    dx, dy, dz = (s * d.length_scale for s in grid.spacing)
    w = np.ones(grid.mesh()[0].shape)  # trapezoidal weights on the node grid
    for ax in range(3):
        sl = [slice(None)] * 3
        sl[ax] = [0, -1]
        w[tuple(sl)] *= 0.5
    stored = np.sum(w * 8000.0 * 500.0 * (res.temperature[-1] - 300.0)) * dx * dy * dz
    injected = 150.0 * t_end
    assert stored == pytest.approx(injected, rel=1e-3)  # observed 5e-5


def test_rejects_unstable_time_step():
    d = DomainConfig()
    grid = Grid.uniform(d, (21, 11, 11))
    dt_max = stable_time_step(grid, HastelloyX(), d, safety=1.0)
    with pytest.raises(ValueError, match="stability"):
        solve_fd(d, LaserConfig(), HastelloyX(), grid, lambda t: (1, 0.5), 1e-4, [1e-4], dt=2 * dt_max)


def test_no_source_stays_at_ambient():
    d = DomainConfig()
    grid = Grid.uniform(d, (11, 6, 6))
    res = solve_fd(d, LaserConfig(power=0.0), HastelloyX(), grid, lambda t: (1, 0.5), 1e-3, [1e-3])
    np.testing.assert_allclose(res.temperature, 300.0)


def test_snapshots_returned_in_time_order():
    d = DomainConfig()
    grid = Grid.uniform(d, (11, 6, 6))
    res = solve_fd(d, LaserConfig(), HastelloyX(), grid, lambda t: (1, 0.5), 2e-3, [2e-3, 1e-3])
    assert list(res.times) == [1e-3, 2e-3]
    assert res.temperature[1].max() > res.temperature[0].max()
