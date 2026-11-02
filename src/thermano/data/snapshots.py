"""Supervised operator-learning datasets stored in HDF5.

Schema (version 1), written by the reference solver or converted from it:

    /inputs   (n_scenarios, n_sensors)  branch input per scenario
    /coords   (n_points, n_coords)      query points shared by all scenarios
    /targets  (n_scenarios, n_points)   quantity of interest (e.g. T or melt-pool depth)
    attrs:    schema_version, units, description, source

The schema is checked on read so that a malformed file fails loudly at load
time rather than as a shape error deep inside training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OperatorDataset:
    inputs: np.ndarray
    coords: np.ndarray
    targets: np.ndarray

    def subset(self, scenarios) -> OperatorDataset:
        idx = np.asarray(scenarios, dtype=int)
        return OperatorDataset(self.inputs[idx], self.coords, self.targets[idx])

    def flatten(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cartesian product scenario x point -> (branch, trunk, target) rows."""
        n_s, n_p = self.targets.shape
        branch = np.repeat(self.inputs, n_p, axis=0)
        trunk = np.tile(self.coords, (n_s, 1))
        return branch, trunk, self.targets.reshape(-1, 1)


def write_dataset(path: str | Path, ds: OperatorDataset, **attrs) -> None:
    with h5py.File(path, "w") as f:
        f["inputs"], f["coords"], f["targets"] = ds.inputs, ds.coords, ds.targets
        f.attrs["schema_version"] = SCHEMA_VERSION
        for k, v in attrs.items():
            f.attrs[k] = v


def read_dataset(path: str | Path) -> OperatorDataset:
    with h5py.File(path, "r") as f:
        version = int(f.attrs.get("schema_version", -1))
        if version != SCHEMA_VERSION:
            raise ValueError(f"{path}: schema_version {version}, expected {SCHEMA_VERSION}")
        ds = OperatorDataset(np.asarray(f["inputs"]), np.asarray(f["coords"]), np.asarray(f["targets"]))
    if ds.inputs.shape[0] != ds.targets.shape[0] or ds.coords.shape[0] != ds.targets.shape[1]:
        raise ValueError(
            f"{path}: inconsistent shapes inputs{ds.inputs.shape} coords{ds.coords.shape} "
            f"targets{ds.targets.shape}"
        )
    return ds
