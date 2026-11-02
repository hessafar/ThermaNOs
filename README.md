# ThermaNO — Thermal Neural Operators for Laser Powder Bed Fusion

[![CI](https://github.com/hessafar/ThermaNO/actions/workflows/ci.yml/badge.svg)](https://github.com/hessafar/ThermaNO/actions/workflows/ci.yml)

ThermaNO builds fast surrogate models of the **3D transient heat equation with a
moving laser source**, the thermal core of laser powder bed fusion (LPBF).
Once trained, a surrogate returns the full temperature field for a new scan
path or laser-power profile in milliseconds, which makes parametric studies and
process optimisation affordable where a finite-difference or finite-element run
per scenario is not.

Two learning modes share one code base:

| Mode | Needs labelled data? | Use when |
|---|---|---|
| **Physics-informed** (`mode: physics`) | No — the heat equation, IC and BCs are the loss | Data are scarce or expensive; single- and multi-track scan paths |
| **Data-driven** (`mode: data`) | Yes — snapshots from the reference solver | High-dimensional inputs (multi-material, power modulation) where the PI loss becomes hard to train |

The physics-informed results are published in
H. Safari, H. Wessels, *Physics-Informed Surrogates for Temperature Prediction of
Multi-Tracks in Laser Powder Bed Fusion*, [arXiv:2502.01820](https://arxiv.org/abs/2502.01820).

## Install

```bash
git clone https://github.com/hessafar/ThermaNO.git && cd ThermaNO
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                     # 53 tests, ~20 s on a laptop CPU; -m "not slow" for the quick subset
```

Exact, tested versions are pinned in `environment.yml`. For HPC, build the
Apptainer image in `containers/thermano.def` (see *Reproducibility*).

## Run

Every run is defined by one YAML file in `configs/`:

```bash
thermano info                                          # versions + registered models/materials
thermano train configs/pi_single_track.yaml            # PI-En-DeepONet, 3 single-track scenarios
thermano train configs/pi_multi_track_sequential.yaml  # sequential PINN, all 6 three-track paths
thermano train configs/pi_single_track.yaml --set training.max_epochs=50   # override any key
thermano reference configs/pi_single_track.yaml --out ref.h5               # FD reference -> HDF5
```

Each run writes to `results/<name>/<config-hash>/`:

```
manifest.json      config, config hash, git commit + dirty flag, package versions, device
train_log.jsonl    loss terms per logged epoch
model.pt           weights + input scalers + the same manifest
metrics.json       MAPE and melt-pool errors against the FD reference
```

## Architecture

```
src/thermano/
├── config.py          typed, validated YAML run configuration (+ hash)
├── registry.py        name -> component registry (models, materials, activations)
├── physics/           heat source, material laws, heat-equation residual  ← shared by FD and PINN
├── models/            FNN, DeepONet, En-DeepONet, PINN (one interface: forward(u, y))
├── data/              scan paths, Sobol sampling, power profiles, HDF5 snapshot schema
├── training/          physics-informed + data losses, one Trainer (Adam / L-BFGS)
├── reference/         explicit finite-difference reference solver
├── evaluation/        MAPE, relative errors, melt-pool geometry
├── pipelines/         physics (operator | sequential) and data-driven workflows
├── provenance.py      seeding, determinism, run manifests, checkpoints
├── optimize.py        surrogate-in-the-loop optimisation (differential evolution)
└── cli.py             `thermano` entry point
```

Design notes and the testing strategy are in [`docs/architecture.md`](docs/architecture.md).

## Testing and validation

* **Unit tests** — models, scalers, sampling, scan-path tree (Eq. 20), config validation.
* **Verification** — the PDE residual is checked against manufactured solutions
  (including the ∇k·∇T term); the FD solver against an analytical decay solution
  (observed spatial order ≈ 2.1), energy conservation and its stability limit.
* **Reproducibility** — same seed and config ⇒ bit-identical weights (CPU).
* **Validation** — `metrics.json` reports MAPE and melt-pool errors against the
  FD reference for every scenario.

CI (GitHub Actions) runs lint and the fast tests on every push, and the full
suite on pull requests to `main`.

## Reproducibility

* Pinned environment (`environment.yml`) and container (`containers/thermano.def`).
* Seeds for Python, NumPy and PyTorch plus deterministic algorithms are set from the config.
* Result folders are named by config hash; manifests record the git commit.
* Datasets follow a versioned HDF5 schema (`thermano/data/snapshots.py`), checked on read.
* SLURM template: `scripts/slurm/train.sbatch`.

## Citing

See `CITATION.cff`. If you use ThermaNO, please cite the arXiv paper above.

## Status

A manuscript on multi-material LPBF optimisation with the data-driven mode is in preparation.
The pre-refactoring research prototype is preserved at tag `v0.1.0`.
