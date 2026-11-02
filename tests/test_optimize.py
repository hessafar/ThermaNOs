import numpy as np

from thermano.optimize import optimize_inputs


def test_recovers_known_optimum():
    """Surrogate stand-in with a known best input: depth(p) = 1 - (p - 0.3)^2."""
    target = 0.95

    def predict(p):
        return 1.0 - (p - 0.3) ** 2

    res = optimize_inputs(predict, lambda p, y: float(np.sum((y - target) ** 2)), [(0.0, 0.25)], seed=1)
    np.testing.assert_allclose(res.x, [0.3 - np.sqrt(0.05)], atol=1e-5)
