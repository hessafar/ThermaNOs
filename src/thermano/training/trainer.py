"""Optimiser loop shared by all modes.

The trainer only knows a model and a loss callable ``loss(model) -> (total, terms)``,
so physics-informed, data-driven and sequential training reuse the same code
path for optimisation, stopping criteria, logging and checkpointing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch


class Trainer:
    def __init__(self, model, loss_fn, cfg, log_path: str | Path | None = None,
                 validate=None) -> None:
        self.model, self.loss_fn, self.cfg = model, loss_fn, cfg
        self.validate = validate  # optional callable(model) -> float, logged each log step
        self.history: list[dict] = []
        self.log_path = Path(log_path) if log_path else None
        params = [p for p in model.parameters() if p.requires_grad]
        if cfg.optimizer == "lbfgs":
            self.optimizer = torch.optim.LBFGS(
                params, lr=cfg.learning_rate, max_iter=25, max_eval=25,
                history_size=50, tolerance_grad=1e-9, tolerance_change=1e-12,
                line_search_fn="strong_wolfe",
            )
        elif cfg.optimizer == "adam":
            self.optimizer = torch.optim.Adam(params, lr=cfg.learning_rate)
        else:  # guarded by config validation; kept for direct API use
            raise ValueError(f"unknown optimizer {cfg.optimizer}")

    def _step(self) -> tuple[float, dict]:
        cache: dict = {}

        def closure():
            self.optimizer.zero_grad()
            total, terms = self.loss_fn(self.model)
            total.backward()
            cache["total"], cache["terms"] = float(total.detach()), terms
            return total

        self.optimizer.step(closure)
        return cache["total"], cache["terms"]

    def fit(self) -> list[dict]:
        prev, start = None, time.perf_counter()
        for epoch in range(1, self.cfg.max_epochs + 1):
            total, terms = self._step()
            if not torch.isfinite(torch.tensor(total)):
                raise FloatingPointError(f"loss became {total} at epoch {epoch}")
            rec = {"epoch": epoch, "loss": total, **terms, "wall_s": time.perf_counter() - start}
            if self.validate and (epoch % self.cfg.log_every == 0 or epoch == 1):
                rec["val"] = self.validate(self.model)
            self.history.append(rec)
            if self.log_path and (epoch % self.cfg.log_every == 0 or epoch == 1):
                with self.log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
            if total < self.cfg.tolerance:
                break
            if prev is not None and abs(prev - total) <= self.cfg.stagnation_rtol * abs(total):
                break
            prev = total
        return self.history
