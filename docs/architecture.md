# ThermaNO architecture

## Goals that shaped the design

1. **One source of truth for the physics.** The heat source, material laws and
   units are implemented once (`thermano.physics`) and used by both the finite-
   difference reference and the physics-informed loss. A surrogate validated
   against the reference therefore cannot disagree with it because of a
   copy-paste difference in the forcing term or material fit.
2. **Experiments are data, not code edits.** A run is fully described by a YAML
   file, parsed into frozen dataclasses and validated before any computation
   (unknown keys, a scan path leaving the domain, an invalid boundary type all
   fail immediately). The config hash names the output folder.
3. **Add, don't modify.** Models, materials and activations are registered by
   name. A new operator architecture (e.g. an FNO) or a new alloy is a new class
   with a decorator; pipelines, trainer and evaluation are untouched.
4. **Every result is traceable.** Each run directory and checkpoint contains a
   manifest: config, hash, git commit and dirty flag, package versions, device.
5. **Testable at small scale.** Every component runs on a laptop CPU with tiny
   settings, so the same code paths used on the GPU cluster are exercised in CI.

## Data flow

```
YAML ──► RunConfig (validated, hashed)
            │
            ├─► data/       scan paths ─► branch input u      collocation points (Sobol)
            │                                                  IC / BC points + normals
            ├─► physics/    material k(T,x), c_p(T) · heat source q(x,t) · residual R
            ├─► models/     DeepONet | En-DeepONet | PINN  : forward(u, y) -> T/T_s
            ├─► training/   loss(model) -> (total, terms)  ·  Trainer(Adam | L-BFGS)
            ├─► reference/  FD solver (same physics/) ─► T_ref
            └─► evaluation/ MAPE, melt-pool length/width/depth ─► metrics.json
```

## Interfaces

* `model(u, y) -> (N, 1)`: `u` branch input (N, m), `y` trunk input (N, 4) = (t, x, y, z).
  Input scaling lives inside the model as buffers, so weights and scaling travel together.
* `loss(model) -> (total, {term: value})`: the trainer is agnostic to which loss it runs.
* `material.conductivity(T, x)`, `material.heat_capacity(T)`: NumPy or torch.

## Units

Model lengths in mm (`length_scale = 1e-3`), time in s, network output
`u = T / T_s` with `T_s = 1800 K`; material properties and power in SI. The
scaled residual is

    du/dt − ∇̂·(k ∇̂u) / (ρ c_p L_s²) − q_v / (ρ c_p T_s) = 0 .

## Testing strategy

| Level | What | Example |
|---|---|---|
| Unit | single functions | scaler maps bounds to [−1, 1]; Eq. 20 node count |
| Verification | numerics vs. mathematics | residual = 0 for manufactured solutions; FD order ≈ 2; energy balance |
| Integration | pipelines end-to-end | tiny PI run writes model, manifest, metrics; data-driven CLI run |
| Reproducibility | determinism | same seed ⇒ identical weights |
| Validation | surrogate vs. reference | MAPE / melt-pool errors in `metrics.json` (paper Figs. 5–8) |

Slow tests (`-m slow`) cover grid-convergence and the sequential strategy and
run on pull requests.

## Known limitations

* Sequential PINN errors can accumulate along long scenarios (each node starts
  from its parent's prediction). Multi-layer builds are future work.
* The physics-informed operator degrades as the number of tracks grows (input
  dimension of the branch net); the data-driven mode is the route for those cases.
* The NumPy FD solver is for verification and small grids; production reference
  data come from the CUDA implementation of the same scheme.
