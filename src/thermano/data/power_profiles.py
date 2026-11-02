"""Laser-power profiles (Gaussian-process samples) used as branch input functions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d


@dataclass(frozen=True)
class PowerProfiles:
    times: np.ndarray  # (n_t,)
    values: np.ndarray  # (n_profiles, n_t)  [W]

    @classmethod
    def from_npz(cls, path: str | Path) -> PowerProfiles:
        with np.load(path) as f:
            return cls(times=np.ravel(f["x_value"]), values=np.atleast_2d(f["y_value"]))

    def __len__(self) -> int:
        return self.values.shape[0]

    def at(self, i: int, t: np.ndarray) -> np.ndarray:
        f = interp1d(self.times, self.values[i], kind="cubic", fill_value="extrapolate")
        return f(np.asarray(t, dtype=float))

    def sensors(self, i: int, duration: float, n_sensors: int) -> np.ndarray:
        return self.at(i, np.linspace(0.0, duration, n_sensors))
