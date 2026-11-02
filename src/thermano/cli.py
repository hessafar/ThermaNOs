"""Command-line entry point.

    thermano train configs/pi_single_track.yaml
    thermano train configs/pi_multi_track_sequential.yaml --set training.max_epochs=50
    thermano reference configs/pi_single_track.yaml      # FD solution only
    thermano info
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import yaml


def _parse_overrides(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        out[key] = yaml.safe_load(value)
    return out


def _setup_dtype(cfg) -> None:
    torch.set_default_dtype(torch.float64 if cfg.dtype == "float64" else torch.float32)


def cmd_train(args) -> int:
    from thermano.config import load_config

    cfg = load_config(args.config, _parse_overrides(args.set))
    _setup_dtype(cfg)
    if cfg.mode == "physics":
        from thermano.pipelines.physics import run
    else:
        from thermano.pipelines.data_driven import run
    report = run(cfg)
    summary = {k: v for k, v in report.items() if k != "per_scenario"}
    print(json.dumps(summary, indent=2))
    return 0


def cmd_reference(args) -> int:
    from thermano.config import load_config
    from thermano.data.snapshots import OperatorDataset, write_dataset
    from thermano.physics.materials import build_material
    from thermano.pipelines.physics import build_scenarios
    from thermano.reference.fd_solver import Grid, solve_fd

    cfg = load_config(args.config, _parse_overrides(args.set))
    grid = Grid.uniform(cfg.domain, cfg.evaluation.grid)
    material = build_material(cfg.material)
    X, Y, Z = grid.mesh()
    inputs, targets, coords = [], [], None
    for sc in build_scenarios(cfg):
        dur = sc.path.duration
        times = [f * dur for f in cfg.evaluation.snapshot_fractions]
        res = solve_fd(cfg.domain, cfg.laser, material, grid, sc.path.center, dur, times, power=sc.power)
        pts = np.concatenate([
            np.column_stack([np.full(X.size, t), X.ravel(), Y.ravel(), Z.ravel()]) for t in res.times
        ])
        coords = pts
        inputs.append(sc.branch)
        targets.append(res.temperature.reshape(-1))
        print(f"scenario {sc.path.order}: {res.n_steps} steps, dt={res.dt:.2e} s, T_max={res.temperature.max():.0f} K")
    write_dataset(args.out, OperatorDataset(np.stack(inputs), coords, np.stack(targets)),
                  units="T [K]; coords (t [s], x/y/z [mm])", source=f"thermano FD, config {cfg.hash()}")
    print(f"wrote {args.out}")
    return 0


def cmd_info(_args) -> int:
    import thermano
    from thermano.registry import ACTIVATIONS, MATERIALS, MODELS

    print(f"thermano {thermano.__version__}  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    print("models:     ", ", ".join(MODELS.names()))
    print("materials:  ", ", ".join(MATERIALS.names()))
    print("activations:", ", ".join(ACTIVATIONS.names()))
    return 0


def main(argv: list[str] | None = None) -> int:
    # importing registers all components
    import thermano.models  # noqa: F401
    import thermano.physics  # noqa: F401

    p = argparse.ArgumentParser(prog="thermano", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train (and evaluate) a surrogate from a YAML config")
    t.add_argument("config")
    t.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set training.max_epochs=50")
    t.set_defaults(func=cmd_train)

    r = sub.add_parser("reference", help="run the FD reference solver and write an HDF5 dataset")
    r.add_argument("config")
    r.add_argument("--out", default="reference.h5")
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    r.set_defaults(func=cmd_reference)

    i = sub.add_parser("info", help="print versions and registered components")
    i.set_defaults(func=cmd_info)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
