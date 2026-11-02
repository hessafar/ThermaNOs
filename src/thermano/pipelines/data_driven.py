"""Data-driven (supervised) operator learning from reference-solver snapshots."""

from __future__ import annotations

import json

import numpy as np
import torch

from thermano.data.snapshots import read_dataset
from thermano.evaluation.metrics import mape, relative_l2
from thermano.models.deeponet import build_model
from thermano.provenance import prepare_run_dir, resolve_device, save_checkpoint, seed_everything
from thermano.training.losses import DataLoss
from thermano.training.trainer import Trainer


def _t(a, device):
    return torch.as_tensor(np.asarray(a), dtype=torch.get_default_dtype(), device=device)


def run(cfg) -> dict:
    run_dir = prepare_run_dir(cfg)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    ds = read_dataset(cfg.data.snapshot_file)
    train, test = ds.subset(cfg.data.train_scenarios), ds.subset(cfg.data.test_scenarios)

    if cfg.data.branch_bounds_low:
        branch_b = (cfg.data.branch_bounds_low, cfg.data.branch_bounds_high)
    else:
        branch_b = (train.inputs.min(0), train.inputs.max(0))
    trunk_b = (ds.coords.min(0), ds.coords.max(0))
    target_scale = float(1.2 * np.abs(train.targets).max())  # headroom for unseen inputs

    b, c, y = train.flatten()
    loss = DataLoss(_t(b, device), _t(c, device), _t(y / target_scale, device))
    model = build_model(cfg.model, ds.inputs.shape[1], branch_b, trunk_b, n_coords=ds.coords.shape[1]).to(device)

    tb, tc, ty = test.flatten()
    tb_t, tc_t = _t(tb, device), _t(tc, device)

    def validate(m):
        with torch.no_grad():
            return relative_l2(m(tb_t, tc_t).cpu().numpy() * target_scale, ty)

    history = Trainer(model, loss, cfg.training, run_dir / "train_log.jsonl", validate).fit()
    save_checkpoint(run_dir / "model.pt", model, cfg, history,
                    {"target_scale": target_scale, "branch_bounds": branch_b, "trunk_bounds": trunk_b})

    with torch.no_grad():
        pred = model(tb_t, tc_t).cpu().numpy() * target_scale
    report = {"test_relative_l2": relative_l2(pred, ty), "test_scenarios": list(cfg.data.test_scenarios)}
    if np.all(np.abs(ty) > 0):
        report["test_mape"] = mape(pred, ty)
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    return report
