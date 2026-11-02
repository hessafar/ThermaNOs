"""Moving hemispherical Gaussian laser heat source (paper Eq. 4).

One implementation is shared by the finite-difference reference solver (NumPy)
and the physics-informed loss (torch), so the surrogate and its reference can
never silently disagree on the forcing term.
"""

from __future__ import annotations

import math

from thermano.physics._backend import xp


def gaussian_heat_source(x, y, z, x_laser, y_laser, power, *, radius, depth, top, length_scale):
    """Volumetric heat input q_v [W/m^3].

    Geometry (x, y, z, radius, depth, top) is in model length units; the result
    is SI because the normalisation constant uses ``radius * length_scale``.
    ``power`` is the absorbed power eta*P [W] and may be an array (power profile).
    """
    m = xp(x, x_laser, power)
    r_si = radius * length_scale
    coeff = 6.0 * math.sqrt(3.0) / (math.pi * math.sqrt(math.pi) * r_si**3)
    planar = ((x - x_laser) ** 2 + (y - y_laser) ** 2) / radius**2
    vertical = (top - z) ** 2 / depth**2
    return power * coeff * m.exp(-3.0 * planar) * m.exp(-3.0 * vertical)
