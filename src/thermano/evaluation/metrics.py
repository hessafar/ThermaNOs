"""Validation metrics used in the paper (Eq. 22) plus melt-pool geometry."""

from __future__ import annotations

import numpy as np


def mape(pred: np.ndarray, ref: np.ndarray) -> float:
    """Mean absolute percentage error, in percent (paper Eq. 22)."""
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    return float(100.0 * np.mean(np.abs((pred - ref) / ref)))


def relative_error(pred: float, ref: float) -> float:
    """Relative error in percent of a scalar (e.g. melt-pool length)."""
    return float(100.0 * abs(pred - ref) / abs(ref))


def relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def melt_pool_dimensions(T: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, T_melt: float) -> dict:
    """Length (x), width (y) and depth (from the top surface) of the region T >= T_melt.

    Edges are located by linear interpolation between the last molten and first
    solid node, so the result is not quantised to the grid spacing.
    Returns zeros if nothing is molten.
    """
    molten = T >= T_melt
    if not molten.any():
        return {"length": 0.0, "width": 0.0, "depth": 0.0}

    def extent(axis_vals, profile):
        idx = np.flatnonzero(profile >= T_melt)
        lo, hi = idx[0], idx[-1]

        def edge(i_in, i_out):
            if i_out < 0 or i_out >= len(axis_vals):
                return axis_vals[i_in]
            a, b = profile[i_in], profile[i_out]
            w = (a - T_melt) / (a - b) if a != b else 0.0
            return axis_vals[i_in] + w * (axis_vals[i_out] - axis_vals[i_in])

        return edge(lo, lo - 1), edge(hi, hi + 1)

    # measure along the three grid lines through the hottest node (on the top
    # surface for a laser melt pool); each line contains at least that node
    i, j, k = np.unravel_index(np.argmax(T), T.shape)
    x0, x1 = extent(x, T[:, j, k])
    y0, y1 = extent(y, T[i, :, k])
    z0, _ = extent(z, T[i, j, :])
    return {"length": float(x1 - x0), "width": float(y1 - y0), "depth": float(z[-1] - z0)}
