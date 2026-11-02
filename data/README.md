# Data

`profile_samples/` — laser-power profiles used as branch-net inputs in the
variable-power experiments. Each `.npz` holds `x_value` (sample times, shape
`(n_t,)`) and `y_value` (power in W, shape `(n_profiles, n_t)`), drawn from a
Gaussian process (`gp_*`) or as sine/modulation families. Load with
`thermano.data.power_profiles.PowerProfiles.from_npz`.

Reference snapshot datasets (HDF5) are not versioned in git because of their
size; they are regenerated with `thermano reference <config>` or converted from
the CUDA solver output into the schema documented in
`src/thermano/data/snapshots.py`.
