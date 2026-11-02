"""Explicit finite-difference reference solver (paper Eqs. 5-6), NumPy version.

This is the *reference* the surrogates are validated against, kept small and
readable on purpose. It shares the heat-source and material code with the
physics-informed loss. The production-scale reference runs use the separate
CUDA implementation of the same scheme (``cuda_laser``, separate repository),
whose HDF5 output is converted to the schema in :mod:`thermano.data.snapshots`; this module is used for
verification tests and small evaluation grids.

Scheme: forward Euler in time, second-order central differences in space,
variable conductivity via  k * lap(T) + grad(k) . grad(T).
Neumann faces: zero normal gradient (mirror ghost nodes). Dirichlet faces: T = T0.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from thermano.physics.heat_source import gaussian_heat_source


@dataclass(frozen=True)
class Grid:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @classmethod
    def uniform(cls, domain, shape: Sequence[int]) -> Grid:
        nx, ny, nz = shape
        return cls(
            np.linspace(0, domain.length, nx),
            np.linspace(0, domain.width, ny),
            np.linspace(0, domain.height, nz),
        )

    @property
    def spacing(self) -> tuple[float, float, float]:
        return (self.x[1] - self.x[0], self.y[1] - self.y[0], self.z[1] - self.z[0])

    def mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.y, self.z, indexing="ij")


@dataclass
class FDResult:
    grid: Grid
    times: np.ndarray  # (n_snap,)
    temperature: np.ndarray  # (n_snap, nx, ny, nz) [K]
    dt: float
    n_steps: int


def stable_time_step(grid: Grid, material, domain, T_max: float = 2500.0, safety: float = 0.9) -> float:
    """Von Neumann limit dt <= 1 / (2 alpha sum 1/dx_i^2), worst case over T in [T0, T_max]."""
    T = np.linspace(domain.initial_temperature, T_max, 512)
    x_probe = np.linspace(0, domain.length, 512)
    k_max = float(np.max(material.conductivity(T, x_probe)))
    rho_cp_min = float(np.min(material.density * material.heat_capacity(T)))
    alpha = k_max / rho_cp_min
    inv = sum(1.0 / (d * domain.length_scale) ** 2 for d in grid.spacing)
    return safety / (2.0 * alpha * inv)


def _padded(a: np.ndarray) -> np.ndarray:
    # mirror ghost nodes => zero normal gradient on every face; Dirichlet faces
    # are overwritten after each step
    return np.pad(a, 1, mode="reflect")


def solve_fd(
    domain,
    laser,
    material,
    grid: Grid,
    laser_center: Callable[[float], Sequence[float]],
    duration: float,
    snapshot_times: Sequence[float],
    power: Callable[[float], float] | None = None,
    initial: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None,
    dt: float | None = None,
    safety: float = 0.9,
) -> FDResult:
    X, Y, Z = grid.mesh()
    ls = domain.length_scale
    dx, dy, dz = (d * ls for d in grid.spacing)
    T = (
        initial(X, Y, Z).astype(float)
        if initial is not None
        else np.full(X.shape, domain.initial_temperature, dtype=float)
    )
    dt_max = stable_time_step(grid, material, domain, safety=1.0)
    if dt is None:
        dt = safety * dt_max
    elif dt > dt_max:
        raise ValueError(f"dt={dt:.3e} s exceeds explicit stability limit {dt_max:.3e} s")
    power = power or (lambda t: laser.power)

    dirichlet = [f for f, kind in domain.boundaries.items() if kind == "dirichlet"]
    face_slices = {
        "x_min": np.s_[0, :, :], "x_max": np.s_[-1, :, :],
        "y_min": np.s_[:, 0, :], "y_max": np.s_[:, -1, :],
        "z_min": np.s_[:, :, 0], "z_max": np.s_[:, :, -1],
    }

    def apply_dirichlet(field):
        for f in dirichlet:
            field[face_slices[f]] = domain.initial_temperature

    apply_dirichlet(T)
    snaps = sorted(snapshot_times)
    out, out_t, t, step = [], [], 0.0, 0
    n_steps = int(np.ceil(duration / dt))
    dt = duration / n_steps  # land exactly on the end time

    def maybe_snapshot():
        while snaps and t >= snaps[0] - 1e-12:
            out.append(T.copy())
            out_t.append(snaps.pop(0))

    maybe_snapshot()
    for step in range(1, n_steps + 1):
        Tp = _padded(T)
        kp = material.conductivity(Tp, _padded(X))
        c = np.s_[1:-1, 1:-1, 1:-1]
        lap = (
            (Tp[2:, 1:-1, 1:-1] - 2 * Tp[c] + Tp[:-2, 1:-1, 1:-1]) / dx**2
            + (Tp[1:-1, 2:, 1:-1] - 2 * Tp[c] + Tp[1:-1, :-2, 1:-1]) / dy**2
            + (Tp[1:-1, 1:-1, 2:] - 2 * Tp[c] + Tp[1:-1, 1:-1, :-2]) / dz**2
        )
        gk_gT = (
            (kp[2:, 1:-1, 1:-1] - kp[:-2, 1:-1, 1:-1]) * (Tp[2:, 1:-1, 1:-1] - Tp[:-2, 1:-1, 1:-1]) / (4 * dx**2)
            + (kp[1:-1, 2:, 1:-1] - kp[1:-1, :-2, 1:-1]) * (Tp[1:-1, 2:, 1:-1] - Tp[1:-1, :-2, 1:-1]) / (4 * dy**2)
            + (kp[1:-1, 1:-1, 2:] - kp[1:-1, 1:-1, :-2]) * (Tp[1:-1, 1:-1, 2:] - Tp[1:-1, 1:-1, :-2]) / (4 * dz**2)
        )
        xl, yl = laser_center(t)
        q = gaussian_heat_source(
            X, Y, Z, xl, yl, power(t),
            radius=laser.radius, depth=laser.penetration_depth,
            top=domain.height, length_scale=ls,
        )
        rho_cp = material.density * material.heat_capacity(T)
        T = T + dt / rho_cp * (kp[c] * lap + gk_gT + q)
        apply_dirichlet(T)
        t = step * dt
        maybe_snapshot()

    return FDResult(grid, np.asarray(out_t), np.stack(out) if out else np.empty(0), dt, n_steps)
