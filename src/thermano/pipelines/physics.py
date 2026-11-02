"""Physics-informed pipelines (no labelled data).

* ``operator``  : one PI-(En)DeepONet learns all scan scenarios at once.
* ``sequential``: one PINN per node of the scenario tree (paper Sec. 2.3); the
  initial condition of each node is the end state of its parent, and a full
  scenario is predicted by chaining the stored node models.

Both are evaluated against the finite-difference reference with the same code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch

from thermano.data.power_profiles import PowerProfiles
from thermano.data.sampling import boundary_points, initial_points, interior_points
from thermano.data.scan_paths import ScanPath, all_scenarios, sensor_bounds, unique_prefixes
from thermano.evaluation.metrics import mape, melt_pool_dimensions, relative_error
from thermano.models.deeponet import build_model
from thermano.physics.materials import build_material
from thermano.provenance import prepare_run_dir, resolve_device, save_checkpoint, seed_everything
from thermano.reference.fd_solver import Grid, solve_fd
from thermano.training.losses import PhysicsInformedLoss, PointSet
from thermano.training.trainer import Trainer


@dataclass
class Scenario:
    path: ScanPath
    branch: np.ndarray  # branch-net input vector
    power: callable  # t -> absorbed power [W]


def build_scenarios(cfg) -> list[Scenario]:
    lz = cfg.laser
    orders = list(lz.scenarios) or all_scenarios(lz.n_tracks)
    if lz.branch_input == "path":
        return [Scenario(ScanPath(lz, o), ScanPath(lz, o).sensor_values(),
                         lambda t, p=lz.power: p + 0.0 * np.asarray(t)) for o in orders]
    profiles = PowerProfiles.from_npz(lz.power_profiles_file)
    path = ScanPath(lz, orders[0])
    n = lz.n_profiles or len(profiles)
    return [Scenario(path, profiles.sensors(i, path.duration, lz.power_sensors),
                     lambda t, i=i: profiles.at(i, t)) for i in range(n)]


def _t(a, device, grad=False):
    return torch.as_tensor(np.asarray(a), dtype=torch.get_default_dtype(), device=device).requires_grad_(grad)


def build_loss(cfg, scenarios, material, device, t_offset=0.0, ic_target=None) -> PhysicsInformedLoss:
    """Assemble interior / IC / BC point sets for all scenarios.

    ``ic_target(coords) -> u0`` overrides the uniform initial temperature
    (used by the sequential strategy to start from the parent's end state).
    """
    d, s = cfg.domain, cfg.sampling
    T0 = d.initial_temperature / d.temperature_scale
    acc = {k: [] for k in ("ib", "ic", "ix", "jb", "jc", "jx", "nb", "nc", "nx", "db", "dc")}
    for i, sc in enumerate(scenarios):
        seed = cfg.seed + 97 * i
        dur = sc.path.track_time if cfg.training.strategy == "sequential" else sc.path.duration
        pts = interior_points(d, dur, s, seed, s.cluster_center)
        centre = sc.path.center(pts[:, 0] + t_offset)
        state = np.column_stack([centre, sc.power(pts[:, 0] + t_offset)])
        acc["ib"].append(np.tile(sc.branch, (len(pts), 1)))
        acc["ic"].append(pts)
        acc["ix"].append(state)

        ic = initial_points(d, s, seed + 1)
        acc["jb"].append(np.tile(sc.branch, (len(ic), 1)))
        acc["jc"].append(ic)
        acc["jx"].append(ic_target(ic) if ic_target else np.full((len(ic), 1), T0))

        for f_i, (face, kind) in enumerate(sorted(d.boundaries.items())):
            bp, nrm = boundary_points(d, dur, face, s.boundary_points_per_face, seed + 10 + f_i)
            key = "n" if kind == "neumann" else "d"
            acc[key + "b"].append(np.tile(sc.branch, (len(bp), 1)))
            acc[key + "c"].append(bp)
            if key == "n":
                acc["nx"].append(nrm)

    cat = {k: np.concatenate(v) if v else None for k, v in acc.items()}
    interior = PointSet(_t(cat["ib"], device), _t(cat["ic"], device, True), _t(cat["ix"], device))
    initial = PointSet(_t(cat["jb"], device), _t(cat["jc"], device), _t(cat["jx"], device))
    neumann = (PointSet(_t(cat["nb"], device), _t(cat["nc"], device, True), _t(cat["nx"], device))
               if cat["nb"] is not None else None)
    dirichlet = PointSet(_t(cat["db"], device), _t(cat["dc"], device)) if cat["db"] is not None else None
    return PhysicsInformedLoss(interior, initial, neumann, dirichlet, material, cfg.laser, cfg.domain,
                               cfg.training.loss_weighting)


def _bounds(cfg, scenarios, duration):
    lo, hi = cfg.domain.bounds()
    trunk = ([0.0, *lo], [duration, *hi])
    if cfg.laser.branch_input == "path":
        b_lo, b_hi = sensor_bounds([s.path for s in scenarios])
    else:
        b = np.stack([s.branch for s in scenarios])
        b_lo, b_hi = b.min(0), b.max(0)
    return (b_lo, b_hi), trunk


@torch.no_grad()
def predict_field(model, branch, grid: Grid, t: float, cfg, device) -> np.ndarray:
    X, Y, Z = grid.mesh()
    coords = np.column_stack([np.full(X.size, t), X.ravel(), Y.ravel(), Z.ravel()])
    u = model(_t(np.tile(branch, (len(coords), 1)), device), _t(coords, device))
    return u.cpu().numpy().reshape(X.shape) * cfg.domain.temperature_scale


# ----------------------------------------------------------------- training


def train_operator(cfg):
    run_dir = prepare_run_dir(cfg)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    material = build_material(cfg.material)
    scenarios = build_scenarios(cfg)
    branch_b, trunk_b = _bounds(cfg, scenarios, scenarios[0].path.duration)
    model = build_model(cfg.model, len(scenarios[0].branch), branch_b, trunk_b).to(device)
    loss = build_loss(cfg, scenarios, material, device)
    history = Trainer(model, loss, cfg.training, run_dir / "train_log.jsonl").fit()
    save_checkpoint(run_dir / "model.pt", model, cfg, history,
                    {"branch_bounds": branch_b, "trunk_bounds": trunk_b, "loss_weights": loss.weights})
    predictors = {i: (lambda t, sc=sc: (model, sc.branch, t)) for i, sc in enumerate(scenarios)}
    return run_dir, model, scenarios, predictors


def train_sequential(cfg):
    """One PINN per scenario-tree node; returns a predictor per full scenario."""
    if cfg.laser.branch_input != "path":
        raise ValueError("sequential strategy is defined over scan paths")
    run_dir = prepare_run_dir(cfg)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    material = build_material(cfg.material)
    lz, d = cfg.laser, cfg.domain
    t_track = lz.track_length / lz.speed
    trunk_b = ([0.0, *d.bounds()[0]], [t_track, *d.bounds()[1]])
    nodes: dict[tuple[int, ...], torch.nn.Module] = {}

    for prefix in unique_prefixes(lz.n_tracks):
        parent = nodes.get(prefix[:-1])
        ic_target = None
        if parent is not None:
            def ic_target(pts, parent=parent):  # end state of the parent node
                end = pts.copy()
                end[:, 0] = t_track
                with torch.no_grad():
                    return parent(_t(np.zeros((len(pts), 1)), device), _t(end, device)).cpu().numpy()

        path = ScanPath(lz, (prefix[-1],))
        sc = Scenario(path, np.zeros(1), lambda t: lz.power + 0.0 * np.asarray(t))
        model = build_model(cfg.model, 1, ([0.0], [1.0]), trunk_b).to(device)
        loss = build_loss(cfg, [sc], material, device, ic_target=ic_target)
        tag = "-".join(map(str, prefix))
        history = Trainer(model, loss, cfg.training, run_dir / f"train_log_{tag}.jsonl").fit()
        save_checkpoint(run_dir / f"node_{tag}.pt", model, cfg, history, {"prefix": prefix})
        nodes[prefix] = model

    scenarios = [Scenario(ScanPath(lz, o), np.zeros(1), lambda t: lz.power + 0.0 * np.asarray(t))
                 for o in (list(lz.scenarios) or all_scenarios(lz.n_tracks))]

    def make_predictor(order):
        def at(t):
            k = min(int(t / t_track), len(order) - 1)
            return nodes[tuple(order[: k + 1])], np.zeros(1), t - k * t_track
        return at

    predictors = {i: make_predictor(sc.path.order) for i, sc in enumerate(scenarios)}
    return run_dir, nodes, scenarios, predictors


# --------------------------------------------------------------- evaluation


def evaluate_against_fd(cfg, run_dir, scenarios, predictors) -> dict:
    device = resolve_device(cfg.device)
    material = build_material(cfg.material)
    ev, d = cfg.evaluation, cfg.domain
    grid = Grid.uniform(d, ev.grid)
    report = {"per_scenario": [], "grid": list(ev.grid)}
    preds_all, refs_all = [], []
    for i, sc in enumerate(scenarios):
        dur = sc.path.duration
        times = [f * dur for f in ev.snapshot_fractions]
        ref = solve_fd(d, cfg.laser, material, grid, lambda t, p=sc.path: p.center(t), dur, times,
                       power=sc.power, safety=ev.fd_cfl_safety)
        rows = []
        for t, T_ref in zip(ref.times, ref.temperature, strict=True):
            model, branch, t_local = predictors[i](min(t, dur * 0.999))
            T_nn = predict_field(model, branch, grid, t_local, cfg, device)
            preds_all.append(T_nn.ravel())
            refs_all.append(T_ref.ravel())
            mp_nn = melt_pool_dimensions(T_nn, grid.x, grid.y, grid.z, ev.melt_temperature)
            mp_fd = melt_pool_dimensions(T_ref, grid.x, grid.y, grid.z, ev.melt_temperature)
            rows.append({
                "t": float(t), "mape": mape(T_nn, T_ref),
                "melt_pool_nn": mp_nn, "melt_pool_fd": mp_fd,
                "melt_pool_rel_err": {k: relative_error(mp_nn[k], mp_fd[k]) if mp_fd[k] else None
                                      for k in mp_fd},
            })
        report["per_scenario"].append({"order": list(sc.path.order), "snapshots": rows})
    report["mape_overall"] = mape(np.concatenate(preds_all), np.concatenate(refs_all))
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def run(cfg) -> dict:
    trainer = train_sequential if cfg.training.strategy == "sequential" else train_operator
    run_dir, _, scenarios, predictors = trainer(cfg)
    if cfg.evaluation.reference == "fd":
        return evaluate_against_fd(cfg, run_dir, scenarios, predictors)
    return {"run_dir": str(run_dir)}
