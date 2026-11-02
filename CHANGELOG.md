# Changelog

## 1.0.0

Restructured the research prototype into an installable, tested package.

* `src/` layout with `pyproject.toml`; `thermano` command-line entry point.
* YAML run configs parsed into validated dataclasses; config hash names result folders.
* One implementation of heat source, materials and residual shared by the FD reference and the PINN loss.
* Model / material / activation registry; DeepONet, En-DeepONet and PINN behind one interface.
* Input scaling stored as model buffers; checkpoints are `state_dict` + manifest (no pickled objects).
* Output transform made explicit in the config (`softplus` | `identity`).
* NumPy FD reference solver with stability check.
* Versioned HDF5 snapshot schema with validation on read; hard-coded paths removed.
* Provenance: seeding, deterministic algorithms, git commit and package versions in every manifest.
* Test suite (unit, verification, integration, reproducibility); GitHub Actions CI; ruff.
* Pinned `environment.yml`, Apptainer definition, SLURM template.

## 0.1.0

Research prototype used for arXiv:2502.01820 (`main.py`, `config.py`, `applications/`).
