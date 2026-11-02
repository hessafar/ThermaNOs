"""Typed, validated run configuration.

Every run is fully described by one YAML file. The YAML is parsed into frozen
dataclasses, validated once at start-up, and its canonical hash is stored with
every checkpoint so that any result can be traced back to its exact inputs.

Units convention (documented here once, used everywhere):
  * lengths in the network/config are in millimetres (``length_scale = 1e-3`` m)
  * time in seconds
  * temperature in the network is ``T / temperature_scale`` (dimensionless)
  * material properties and laser power are SI
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass(frozen=True)
class DomainConfig:
    length: float = 2.0  # x [mm]
    width: float = 1.0  # y [mm]
    height: float = 1.0  # z [mm]
    length_scale: float = 1.0e-3  # metres per model length unit
    temperature_scale: float = 1800.0  # K
    initial_temperature: float = 300.0  # K
    # boundary type per face: "neumann" (insulated) or "dirichlet" (T = T0)
    boundaries: dict[str, str] = field(
        default_factory=lambda: {
            "x_min": "neumann", "x_max": "neumann",
            "y_min": "neumann", "y_max": "neumann",
            "z_min": "dirichlet", "z_max": "neumann",
        }
    )

    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return (0.0, 0.0, 0.0), (self.length, self.width, self.height)


@dataclass(frozen=True)
class LaserConfig:
    power: float = 150.0  # effective power eta*P [W]
    radius: float = 0.45  # [mm]
    penetration_depth: float = 0.5  # [mm]
    speed: float = 100.0  # [mm/s]
    track_length: float = 1.0  # [mm]
    hatch_spacing: float = 0.25  # [mm] distance between neighbouring tracks
    start: tuple[float, float] = (0.5, 0.5)  # first track start point [mm]
    scan_direction: tuple[float, float] = (1.0, 0.0)
    jump_direction: tuple[float, float] = (0.0, 1.0)
    n_tracks: int = 1
    sensors_per_track: int = 20  # time samples of the path seen by the branch net
    # track orders to learn; empty = all n_tracks! permutations
    scenarios: tuple[tuple[int, ...], ...] = ()
    # branch input: the scan "path" or a time-varying laser "power" profile
    branch_input: Literal["path", "power"] = "path"
    power_profiles_file: str = ""
    n_profiles: int = 0
    power_sensors: int = 50


@dataclass(frozen=True)
class MaterialConfig:
    type: str = "hastelloy_x"
    density: float = 8351.91  # kg/m^3
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    type: Literal["deeponet", "en_deeponet", "pinn"] = "en_deeponet"
    branch_layers: tuple[int, ...] = (64, 64)
    trunk_layers: tuple[int, ...] = (64, 64, 64, 64, 64)
    latent_dim: int = 64
    activation: str = "sin"
    output_transform: Literal["softplus", "identity"] = "softplus"


@dataclass(frozen=True)
class SamplingConfig:
    num_time_slices: int = 50
    points_per_slice: int = 400
    initial_points: int = 4000
    boundary_points_per_face: int = 2000
    cluster_fraction: float = 0.5  # share of extra points clustered around the laser path
    cluster_width: float = 0.3
    cluster_gamma: float = 2.0
    cluster_center: float | None = None  # x-location to refine (laser track / interface)


@dataclass(frozen=True)
class TrainingConfig:
    # "operator": one PI-DeepONet for all scenarios; "sequential": one PINN per scenario-tree node
    strategy: Literal["operator", "sequential"] = "operator"
    optimizer: Literal["lbfgs", "adam"] = "lbfgs"
    learning_rate: float = 1.0
    max_epochs: int = 1000
    tolerance: float = 1.0e-6
    stagnation_rtol: float = 1.0e-15
    log_every: int = 10
    loss_weighting: Literal["initial_loss", "fixed"] = "initial_loss"


@dataclass(frozen=True)
class EvaluationConfig:
    reference: Literal["fd", "none"] = "fd"
    grid: tuple[int, int, int] = (41, 21, 21)
    snapshot_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    melt_temperature: float = 1658.0  # K (solidus)
    fd_cfl_safety: float = 0.9


@dataclass(frozen=True)
class DataConfig:
    """Only used in ``mode: data`` (supervised operator learning)."""

    snapshot_file: str = ""
    inputs_file: str = ""
    train_scenarios: tuple[int, ...] = ()
    test_scenarios: tuple[int, ...] = ()
    branch_bounds_low: tuple[float, ...] = ()
    branch_bounds_high: tuple[float, ...] = ()


@dataclass(frozen=True)
class RunConfig:
    name: str
    mode: Literal["physics", "data"]
    seed: int = 4000
    device: str = "auto"
    dtype: Literal["float32", "float64"] = "float32"
    output_dir: str = "results"
    domain: DomainConfig = field(default_factory=DomainConfig)
    laser: LaserConfig = field(default_factory=LaserConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # ------------------------------------------------------------------ utils
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        """Canonical SHA-256 of the configuration (first 12 hex chars)."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=list).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def validate(self) -> RunConfig:
        errors: list[str] = []
        d, lz = self.domain, self.laser
        if min(d.length, d.width, d.height) <= 0:
            errors.append("domain dimensions must be positive")
        if lz.radius <= 0 or lz.speed <= 0 or lz.power < 0:
            errors.append("laser radius/speed must be > 0 and power >= 0")
        x_end = lz.start[0] + lz.scan_direction[0] * lz.track_length
        y_end = lz.start[1] + lz.jump_direction[1] * lz.hatch_spacing * (lz.n_tracks - 1)
        if not (0 <= x_end <= d.length and 0 <= y_end <= d.width):
            errors.append(f"scan path leaves the domain (end point {x_end:.3f}, {y_end:.3f})")
        faces = {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}
        if set(d.boundaries) != faces or not set(d.boundaries.values()) <= {"neumann", "dirichlet"}:
            errors.append(f"domain.boundaries must give 'neumann'|'dirichlet' for {sorted(faces)}")
        if not 0.0 <= self.sampling.cluster_fraction <= 1.0:
            errors.append("sampling.cluster_fraction must be in [0, 1]")
        for sc in lz.scenarios:
            if len(sc) != len(lz.scenarios[0]) or len(set(sc)) != len(sc) or max(sc) >= lz.n_tracks:
                errors.append(f"laser.scenarios entry {sc} is not a valid track order")
        if lz.branch_input == "power" and not lz.power_profiles_file:
            errors.append("branch_input 'power' requires laser.power_profiles_file")
        if self.mode == "data" and not self.data.snapshot_file:
            errors.append("mode 'data' requires data.snapshot_file")
        if errors:
            raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))
        return self


def _build(cls: type, raw: dict[str, Any] | None) -> Any:
    raw = raw or {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        default = cls.__dataclass_fields__[name].default_factory  # type: ignore[attr-defined]
        if callable(default) and is_dataclass(default):
            kwargs[name] = _build(default, value)
        elif isinstance(value, list):
            kwargs[name] = tuple(tuple(v) if isinstance(v, list) else v for v in value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> RunConfig:
    """Load a YAML run file, apply dotted ``overrides`` and validate."""
    raw = yaml.safe_load(Path(path).read_text())
    for dotted, value in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return _build(RunConfig, raw).validate()
