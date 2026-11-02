"""Scan paths: laser position over time and the branch-net encoding of a path.

A *scenario* is an ordering of the parallel tracks (e.g. 2 -> 0 -> 1). With
``n`` tracks there are ``n!`` scenarios; the sequential-PINN strategy trains one
network per unique scenario *prefix*, of which there are
``sum_{i=0}^{n-1} prod_{j=0}^{i} (n - j)`` (paper Eq. 20).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScanPath:
    laser: object  # LaserConfig
    order: tuple[int, ...]

    def __post_init__(self) -> None:
        if sorted(self.order) != sorted(set(self.order)) or max(self.order) >= self.laser.n_tracks:
            raise ValueError(f"invalid track order {self.order} for {self.laser.n_tracks} tracks")

    @property
    def track_time(self) -> float:
        return self.laser.track_length / self.laser.speed

    @property
    def duration(self) -> float:
        return len(self.order) * self.track_time

    def _track_start(self, track: int) -> np.ndarray:
        lz = self.laser
        return np.asarray(lz.start) + track * lz.hatch_spacing * np.asarray(lz.jump_direction)

    def center(self, t: np.ndarray | float) -> np.ndarray:
        """Laser centre (x, y) at time(s) t; shape (..., 2)."""
        t = np.clip(np.asarray(t, dtype=float), 0.0, self.duration)
        idx = np.minimum((t / self.track_time).astype(int), len(self.order) - 1)
        starts = np.stack([self._track_start(self.order[i]) for i in range(len(self.order))])
        local = t - idx * self.track_time
        return starts[idx] + (self.laser.speed * local)[..., None] * np.asarray(self.laser.scan_direction)

    def sensor_values(self) -> np.ndarray:
        """Branch input: laser (x, y) at ``sensors_per_track + 1`` fixed instants per track."""
        k = self.laser.sensors_per_track
        local = np.linspace(0.0, self.track_time, k + 1)
        pts = [
            self._track_start(tr) + self.laser.speed * local[:, None] * np.asarray(self.laser.scan_direction)
            for tr in self.order
        ]
        return np.concatenate(pts).reshape(-1)


def all_scenarios(n_tracks: int) -> list[tuple[int, ...]]:
    return list(itertools.permutations(range(n_tracks)))


def unique_prefixes(n_tracks: int) -> list[tuple[int, ...]]:
    """Nodes of the scenario tree (paper Fig. 3b); one sequential PINN per node."""
    return [p for r in range(1, n_tracks + 1) for p in itertools.permutations(range(n_tracks), r)]


def sequential_pinn_count(n_tracks: int) -> int:
    """Closed form of paper Eq. 20."""
    return sum(math.prod(n_tracks - j for j in range(i + 1)) for i in range(n_tracks))


def sensor_bounds(paths: Sequence[ScanPath]) -> tuple[np.ndarray, np.ndarray]:
    s = np.stack([p.sensor_values() for p in paths])
    return s.min(axis=0), s.max(axis=0)
