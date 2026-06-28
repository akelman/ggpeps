

# Changelog

## Unreleased

### Breaking changes

#### Generic Z2 configuration

The standard two-dimensional Z2 ansatz is now routed through the generic configuration class:

- `src/ggpeps/system/config_Z2.py`
- `Z2System2D_Config`

This config replaces the old G4-specific manager path and supports `ncopy=1` and even `ncopy` values at the level of the parameterization. The manager now constructs Z2 systems by passing `ncopy` explicitly from the command line.

The generic config has been validated against the latest `dev` branch for `ncopy=1`, `ncopy=2`, and `ncopy=4` using equivalent parameter vectors and equivalent legacy conventions where needed. The `ncopy=8` path remains blocked because generic config construction becomes very expensive inside `generate_gauged_projector_terms()` before exact evaluation, Monte Carlo evaluation, or minimization begins.

#### Parameter-order compatibility

The generic Z2 config uses a systematic parameter order generated from `ncopy`. Saved parameter vectors from older config classes should not be assumed to be index-compatible with `Z2System2D_Config`.

When comparing against older results or loading old saved parameters, translate parameters by name/order rather than assuming that raw array indices match. `manager.py` now provides the `--input_param_order` compatibility flag for dev-vs-z2 regression checks.

Supported legacy parameter-order modes are:

- `current`: use the active config order without translation.
- `legacy_1c`: interpret parameters using the old one-copy pure-gauge order `[tr, yr, zr, ti, yi, zi]` and translate to the generic `ncopy=1` order.
- `legacy_g2c_f2c`: interpret parameters using the old G2C/F2C order and translate to the generic `ncopy=2` order.
- `legacy_g4c_f4c`: interpret parameters using the old G4C/F4C order. This order is index-compatible with the generic `ncopy=4` order, but the flag records the intended legacy interpretation.

For `ncopy = n`, the generic Z2 parameter order is:

```text
real part:
    t1r, ..., tnr,
    y1r, ..., ynr,
    z1r, ..., znr,
    a12r, b12r, c12r, d12r,
    a13r, b13r, c13r, d13r,
    ...,
    a(n-1)nr, b(n-1)nr, c(n-1)nr, d(n-1)nr

imaginary part:
    t1i, ..., tni,
    y1i, ..., yni,
    z1i, ..., zni,
    a12i, b12i, c12i, d12i,
    a13i, b13i, c13i, d13i,
    ...,
    a(n-1)ni, b(n-1)ni, c(n-1)ni, d(n-1)ni
```

Use name-based reordering utilities rather than manual index assumptions when migrating old parameter vectors.

#### Parameter-constraint compatibility

The generic Z2 config has a `param_constraints` option that controls zeroing rules before evaluation or minimization.

The default mode is:

- `current`: use the generic Z2 zeroing rules.

For regression checks against the old four-copy config, `manager.py` also provides:

- `legacy_g4c_f4c`: preserve the legacy G4C/F4C fermionic-layer zeroing rules.

The legacy G4C/F4C mode is required when comparing generic `ncopy=4` runs to the old `dev` G4C/F4C config, because the old config did not zero any `t_i` parameters in fermionic layers, while the current generic convention zeros the even-labelled `t` copies in fermionic layers under U(1) symmetry.

Use both flags together for four-copy legacy comparisons:

```bash
python manager.py <mode> Z2 \
  --ncopy 4 \
  --input_param_order legacy_g4c_f4c \
  --param_constraints legacy_g4c_f4c
```

#### Fixed-copy configs and `ncopy`

The fixed-copy configs now accept an explicit `ncopy` keyword argument and validate that it matches the class-specific copy number:

- `D6System2D_Config` expects `ncopy=2`.
- `Z2System2D_2col_Config` expects `ncopy=2`.
- `Z2System2D_2col_1copy_Config` expects `ncopy=1`.

This allows `manager.py` to pass `ncopy=args.ncopy` through a shared config-construction path while still producing explicit errors for unsupported copy numbers.

### Validation

The generic Z2 refactor was validated against `dev` commit `3679276ce7fe78f781118393d22435016609136a` using this branch at commit `b7b44b0c2b6924271ea3c103711c89096ab59880`. The regression checks used identical seeds, couplings, lattice size, layer counts, and equivalent parameter vectors.

Passed regression checks:

- `eval-exact`, `ncopy=1`, using `--input_param_order legacy_1c`.
- `eval-exact`, `ncopy=2`, using `--input_param_order legacy_g2c_f2c`.
- `eval-exact`, `ncopy=4`, using `--input_param_order legacy_g4c_f4c --param_constraints legacy_g4c_f4c`.
- `eval-mc`, `ncopy=1`, using `--input_param_order legacy_1c`.
- `eval-mc`, `ncopy=2`, using `--input_param_order legacy_g2c_f2c`.
- `eval-mc`, `ncopy=4`, using `--input_param_order legacy_g4c_f4c --param_constraints legacy_g4c_f4c`.
- `min-exact`, `ncopy=1`, using `--input_param_order legacy_1c`.
- `min-exact`, `ncopy=2`, using `--input_param_order legacy_g2c_f2c`.
- `min-exact`, `ncopy=4`, using `--input_param_order legacy_g4c_f4c --param_constraints legacy_g4c_f4c`.

All comparable `ncopy=1`, `ncopy=2`, and `ncopy=4` dev-vs-z2 regression rows passed. Differences, where present, were roundoff-level only. The `ncopy=8` row remains blocked/postponed because generic `ncopy=8` config construction does not currently complete in practical time.