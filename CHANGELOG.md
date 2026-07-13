

# Changelog

## Unreleased

### Breaking changes

#### Generic Z2 configuration

The standard two-dimensional Z2 ansatz is now routed through the generic configuration class:

- `src/ggpeps/system/config_Z2.py`
- `Z2System2D_Config`

This config replaces the old G4-specific manager path and supports `ncopy=1` and even `ncopy` values at the level of the parameterization. The manager now constructs Z2 systems by passing `ncopy` explicitly from the command line.

The generic config was validated against `dev` commit `3679276ce7fe78f781118393d22435016609136a`. The current branch state after removing the test-only constraint compatibility layer is `c9f62dc0be6d91fb12ca3ea46205283446d72d39`.

Regression checks were run for `ncopy=1`, `ncopy=2`, and `ncopy=4`. The `ncopy=1` and `ncopy=2` legacy comparisons are supported through parameter-order translation. The old four-copy G4C/F4C constraint convention was also tested during validation, but exact compatibility with that convention required a temporary `param_constraints` runtime flag. That flag was removed because it was too invasive for a test-only compatibility path: it leaked into `manager.py`, the generic Z2 config, and unrelated config constructors. The generic `ncopy=4` ansatz now keeps only the current U(1)-symmetric zeroing convention.

The `ncopy=8` path remains blocked/postponed because generic config construction becomes very expensive inside `generate_gauged_projector_terms()` before exact evaluation, Monte Carlo evaluation, or minimization begins.

#### Parameter-order compatibility

The generic Z2 config uses a systematic parameter order generated from `ncopy`. Saved parameter vectors from older config classes should not be assumed to be index-compatible with `Z2System2D_Config`.

When comparing against older results or loading old saved parameters, translate parameters by name/order rather than assuming that raw array indices match. `manager.py` provides the `--legacy_param_order` flag for this purpose. The legacy order is selected from `--ncopy`.

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

#### Fixed-copy configs and `ncopy`

The fixed-copy configs now accept an explicit `ncopy` keyword argument and validate that it matches the class-specific copy number:

- `D6System2D_Config` expects `ncopy=2`.
- `Z2System2D_2col_Config` expects `ncopy=2`.
- `Z2System2D_2col_1copy_Config` expects `ncopy=1`.

This allows `manager.py` to pass `ncopy=args.ncopy` through a shared config-construction path while still producing explicit errors for unsupported copy numbers.

### Validation

Validation was performed against `dev` commit `3679276ce7fe78f781118393d22435016609136a`. The current branch state after reverting the test-only constraint compatibility layer is `c9f62dc0be6d91fb12ca3ea46205283446d72d39`.

The following regression comparisons were run with identical seeds, couplings, lattice sizes, layer counts, and equivalent parameter vectors:

- `ncopy=1`: legacy one-copy pure-gauge comparisons passed using `--ncopy 1 --legacy_param_order`.
- `ncopy=2`: legacy G2C/F2C comparisons passed using `--ncopy 2 --legacy_param_order`.
- `ncopy=4`: legacy G4C/F4C comparisons were tested. Exact reproduction of the old four-copy constraint convention required a temporary `param_constraints` compatibility flag.

Additional `L=4` Monte Carlo comparisons were run with short statistics for `ncopy=1`, `ncopy=2`, and `ncopy=4` during validation. The `ncopy=1` and `ncopy=2` checks remain supported by the current code through parameter-order translation.

The temporary `param_constraints` flag and the associated legacy G4C/F4C constraint mode were removed after validation because the implementation was too invasive for a test-only compatibility mechanism. It required changes to the runtime CLI, the generic Z2 config, and unrelated config constructors. The current code therefore does not support exact legacy G4C/F4C constraint compatibility; generic `ncopy=4` uses the current U(1)-symmetric zeroing convention.

Current status:

- `ncopy=1`: supported and validated against the legacy one-copy convention.
- `ncopy=2`: supported and validated against the legacy G2C/F2C convention.
- `ncopy=4`: supported as the current generic four-copy ansatz; exact legacy G4C/F4C constraint compatibility is intentionally not supported.
- `ncopy=8`: blocked/postponed during generic config construction.
