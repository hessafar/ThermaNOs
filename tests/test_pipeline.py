"""End-to-end smoke and reproducibility tests (small enough for CI)."""

import json

import numpy as np
import pytest
import torch

from thermano.cli import main
from thermano.data.snapshots import OperatorDataset, write_dataset
from thermano.pipelines import physics


def test_physics_pipeline_runs_and_writes_artifacts(tiny_cfg):
    report = physics.run(tiny_cfg)
    run_dir = physics.prepare_run_dir(tiny_cfg)
    assert (run_dir / "model.pt").exists() and (run_dir / "metrics.json").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["config_hash"] == tiny_cfg.hash()
    assert np.isfinite(report["mape_overall"])
    ckpt = torch.load(run_dir / "model.pt", weights_only=False)
    assert ckpt["manifest"]["config"]["seed"] == tiny_cfg.seed


def test_training_reduces_loss(tiny_cfg):
    _, _, _, _ = physics.train_operator(tiny_cfg)
    log = [json.loads(line) for line in (physics.prepare_run_dir(tiny_cfg) / "train_log.jsonl").open()]
    assert log[-1]["loss"] < log[0]["loss"]


def test_same_seed_same_result(tiny_cfg):
    _, m1, _, _ = physics.train_operator(tiny_cfg)
    _, m2, _, _ = physics.train_operator(tiny_cfg)
    for (k, a), (_, b) in zip(m1.state_dict().items(), m2.state_dict().items(), strict=True):
        assert torch.equal(a, b), k


@pytest.mark.slow
def test_sequential_strategy_builds_all_nodes(tiny_cfg):
    from dataclasses import replace

    cfg = replace(
        tiny_cfg,
        laser=replace(tiny_cfg.laser, n_tracks=2, hatch_spacing=0.2, scenarios=()),
        model=replace(tiny_cfg.model, type="pinn"),
        training=replace(tiny_cfg.training, strategy="sequential", max_epochs=3),
    ).validate()
    run_dir, nodes, scenarios, _ = physics.train_sequential(cfg)
    assert set(nodes) == {(0,), (1,), (0, 1), (1, 0)}
    assert len(scenarios) == 2 and len(list(run_dir.glob("node_*.pt"))) == 4


def test_data_driven_cli(tmp_path):
    rng = np.random.default_rng(0)
    inputs = rng.random((6, 3))
    coords = np.linspace(0, 1, 20)[:, None]
    targets = 1.0 + inputs[:, :1] * np.sin(3 * coords.T)  # known smooth operator
    h5 = tmp_path / "d.h5"
    write_dataset(h5, OperatorDataset(inputs, coords, targets))
    cfg = tmp_path / "dd.yaml"
    cfg.write_text(
        f"name: dd\nmode: data\ndtype: float64\noutput_dir: {tmp_path}\n"
        f"data: {{snapshot_file: {h5}, train_scenarios: [0,1,2,3,4], test_scenarios: [5]}}\n"
        "model: {branch_layers: [16], trunk_layers: [16], latent_dim: 8}\n"
        "training: {optimizer: adam, learning_rate: 0.01, max_epochs: 30}\n"
        "evaluation: {reference: none}\n"
    )
    assert main(["train", str(cfg)]) == 0
    metrics = json.loads(next(tmp_path.glob("dd/*/metrics.json")).read_text())
    assert metrics["test_relative_l2"] < 1.0
