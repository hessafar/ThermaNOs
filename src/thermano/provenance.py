"""Reproducibility: seeding, environment capture and run manifests.

Every run directory gets ``manifest.json`` recording *what* produced it (config
and its hash, git commit and dirty flag, package versions, device). Every
checkpoint embeds the same manifest, so a weights file alone is enough to
reconstruct how it was made.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import thermano


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - also seed any legacy global-RNG use
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _git(*args: str) -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def manifest(cfg) -> dict:
    return {
        "thermano_version": thermano.__version__,
        "config_hash": cfg.hash(),
        "config": cfg.to_dict(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": _git("status", "--porcelain") not in ("", "unknown"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "device": str(resolve_device(cfg.device)),
        "host": platform.node(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def prepare_run_dir(cfg) -> Path:
    """results/<name>/<config-hash>/ — re-running the same config targets the same folder."""
    run_dir = Path(cfg.output_dir) / cfg.name / cfg.hash()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest(cfg), indent=2, default=list))
    return run_dir


def save_checkpoint(path: Path, model, cfg, history, extra: dict | None = None) -> None:
    torch.save(
        {"state_dict": model.state_dict(), "manifest": manifest(cfg), "history": history,
         **(extra or {})},
        path,
    )
