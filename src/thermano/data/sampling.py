"""Collocation-point sampling for the physics-informed loss.

* Scrambled Sobol (low-discrepancy) points in space, uniform slices in time.
* An optional share of extra points is confined in x to the band
  [center - width, center + width] by a tanh map (denser towards the band
  edges as gamma grows), to resolve steep gradients around the laser track or
  a material interface.
* All samplers are seeded explicitly: identical config => identical points.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import qmc

FACES = {
    "x_min": (0, 0), "x_max": (0, 1),
    "y_min": (1, 0), "y_max": (1, 1),
    "z_min": (2, 0), "z_max": (2, 1),
}


def _sobol(n: int, d: int, seed: int) -> np.ndarray:
    with warnings.catch_warnings():
        # n is set by the physics (slices x points), not always a power of two
        warnings.filterwarnings("ignore", message=".*balance properties.*")
        return qmc.Sobol(d=d, scramble=True, seed=seed).random(n)


def cluster_transform(xi: np.ndarray, center: float, width: float, gamma: float) -> np.ndarray:
    """Map uniform xi in [0, 1] into the band ``center +/- width``."""
    return center + width * np.tanh(gamma * (2.0 * xi - 1.0))


def interior_points(domain, duration: float, sampling, seed: int, cluster_center: float | None = None):
    """(N, 4) array of (t, x, y, z) collocation points."""
    lo, hi = np.asarray(domain.bounds()[0]), np.asarray(domain.bounds()[1])
    nt, ppt = sampling.num_time_slices, sampling.points_per_slice
    times = np.linspace(0.0, 0.999 * duration, nt)

    space = qmc.scale(_sobol(nt * ppt, 3, seed), lo, hi)
    pts = [np.column_stack([np.repeat(times, ppt), space])]

    n_cl = int(sampling.cluster_fraction * ppt)
    if n_cl and cluster_center is not None:
        raw = _sobol(nt * n_cl, 3, seed + 1)
        cl = qmc.scale(raw, lo, hi)
        cl[:, 0] = np.clip(
            cluster_transform(raw[:, 0], cluster_center, sampling.cluster_width, sampling.cluster_gamma),
            lo[0], hi[0],
        )
        pts.append(np.column_stack([np.repeat(times, n_cl), cl]))
    return np.vstack(pts)


def initial_points(domain, sampling, seed: int) -> np.ndarray:
    lo, hi = domain.bounds()
    space = qmc.scale(_sobol(sampling.initial_points, 3, seed), lo, hi)
    return np.column_stack([np.zeros(len(space)), space])


def boundary_points(domain, duration: float, face: str, n: int, seed: int):
    """Points on one face and its outward normal, both (N, 4) with t-component 0 in the normal."""
    axis, side = FACES[face]
    lo, hi = np.asarray(domain.bounds()[0]), np.asarray(domain.bounds()[1])
    raw = _sobol(n, 4, seed)
    pts = qmc.scale(raw, np.r_[0.0, lo], np.r_[duration, hi])
    pts[:, axis + 1] = hi[axis] if side else lo[axis]
    normal = np.zeros_like(pts)
    normal[:, axis + 1] = 1.0 if side else -1.0
    return pts, normal
