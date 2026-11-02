"""Surrogate-in-the-loop process optimisation.

The point of a fast surrogate: an optimiser can call it thousands of times.
``optimize_inputs`` searches the branch-input space (e.g. laser-power modulation
parameters) for the input that minimises a user objective on the prediction.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import differential_evolution


def optimize_inputs(
    predict: Callable[[np.ndarray], np.ndarray],
    objective: Callable[[np.ndarray, np.ndarray], float],
    bounds: Sequence[tuple[float, float]],
    seed: int = 0,
    maxiter: int = 200,
):
    """Global search with differential evolution; returns scipy's OptimizeResult."""

    def f(params):
        return objective(params, predict(np.asarray(params)))

    return differential_evolution(f, bounds=list(bounds), seed=seed, maxiter=maxiter, polish=True, tol=1e-8)
