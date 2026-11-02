"""Verification of the physics kernels shared by the FD solver and the PINN loss."""

import math

import numpy as np
import pytest
import torch
from scipy import integrate

from thermano.config import DomainConfig, LaserConfig
from thermano.physics.heat_equation import heat_equation_residual
from thermano.physics.heat_source import gaussian_heat_source
from thermano.physics.materials import ConstantMaterial, GradedConductivity, HastelloyX


@pytest.mark.parametrize("depth", [0.45, 0.6])
def test_heat_source_total_power(depth):
    """Volume integral over the half-space equals P * c / r (exactly P when c = r)."""
    r, P, ls = 0.45, 150.0, 1e-3

    def q(z, y, x):  # x, y, z in mm; top surface at z = 0
        return gaussian_heat_source(x, y, z, 0.0, 0.0, P, radius=r, depth=depth, top=0.0, length_scale=ls)

    L = 6 * r
    val, _ = integrate.tplquad(q, -L, L, -L, L, -3 * depth, 0.0, epsabs=1e-6)
    assert val * ls**3 == pytest.approx(P * depth / r, rel=1e-4)


def test_heat_source_numpy_torch_agree():
    args = dict(radius=0.45, depth=0.5, top=1.0, length_scale=1e-3)
    pts = np.random.default_rng(0).random((50, 3))
    a = gaussian_heat_source(*pts.T, 0.5, 0.5, 150.0, **args)
    b = gaussian_heat_source(*torch.tensor(pts).T, 0.5, 0.5, torch.tensor(150.0), **args)
    np.testing.assert_allclose(a, b.numpy(), rtol=1e-12)


def test_hastelloy_matches_paper_eq21():
    m = HastelloyX()
    T = np.array([300.0, 1000.0, 1816.8])
    k_expected = 229.87 + 0.0184 * T + 225.10 * np.tanh(0.0118 * (T - 1816.8))
    np.testing.assert_allclose(m.conductivity(T), k_expected)
    # apparent heat capacity peaks around the melting range (latent heat)
    T_scan = np.linspace(300, 2500, 2201)
    assert 1750 < T_scan[np.argmax(m.heat_capacity(T_scan))] < 1880
    assert np.all(m.conductivity(T_scan) > 0) and np.all(m.heat_capacity(T_scan) > 0)


def test_graded_conductivity_limits():
    m = GradedConductivity(k_out=25.0, k_in=7.5, x1=0.8, x2=1.2, beta=200.0)
    T = np.full(3, 500.0)
    np.testing.assert_allclose(m.conductivity(T, np.array([0.0, 1.0, 2.0])), [25.0, 7.5, 25.0], rtol=1e-6)


def _laser_off():
    return LaserConfig(power=0.0)


def test_residual_vanishes_for_exact_solution():
    """Method of manufactured solutions: u = T0 + A exp(-a pi^2 t / L^2) cos(pi x / L)."""
    d = DomainConfig(length=2.0)
    mat = ConstantMaterial(density=8000.0, conductivity_value=20.0, heat_capacity_value=500.0)
    a_hat = 20.0 / (8000.0 * 500.0 * d.length_scale**2)  # diffusivity in mm^2/s
    coords = torch.rand(200, 4, requires_grad=True)
    with torch.no_grad():
        coords[:, 0] *= 1e-2
        coords[:, 1] *= d.length
    t, x = coords[:, 0], coords[:, 1]
    u = (0.2 + 0.1 * torch.exp(-a_hat * math.pi**2 * t / d.length**2) * torch.cos(math.pi * x / d.length))
    res = heat_equation_residual(u[:, None], coords, torch.zeros(200, 3), mat, _laser_off(), d)
    scale = (a_hat * math.pi**2 / d.length**2) * 0.1
    assert res.abs().max() / scale < 1e-10


def test_residual_includes_conductivity_gradient_term():
    """For k = k0 + k1*T and u linear in x, div(k grad T) = k1 * (T_s c1)^2 * ... (the grad k . grad T term)."""

    class LinearK(ConstantMaterial):
        def conductivity(self, T, x=None):
            return 10.0 + 0.01 * T

    d = DomainConfig()
    mat = LinearK(density=8000.0, conductivity_value=0.0, heat_capacity_value=500.0)
    c0, c1 = 0.3, 0.2
    coords = torch.rand(50, 4, requires_grad=True)
    u = c0 + c1 * coords[:, 1:2]
    res = heat_equation_residual(u, coords, torch.zeros(50, 3), mat, _laser_off(), d)
    Ts, ls = d.temperature_scale, d.length_scale
    expected = -(0.01 * Ts * c1 * c1) / (8000.0 * 500.0 * ls**2)
    assert torch.allclose(res, torch.full_like(res, expected), rtol=1e-10)


def test_residual_source_term_sign():
    """A cold, uniform field under the laser must have negative residual (heating is missing)."""
    d, lz = DomainConfig(), LaserConfig()
    coords = torch.tensor([[0.0, 1.0, 0.5, d.height]], requires_grad=True)
    u = torch.full((1, 1), 0.2) + 0.0 * coords.sum()
    state = torch.tensor([[1.0, 0.5, lz.power]])
    res = heat_equation_residual(u, coords, state, HastelloyX(), lz, d)
    assert res.item() < 0
