

# Changelog

## Unreleased

### Breaking changes

#### Generic Z2 configuration

The standard two-dimensional Z2 ansatz is now routed through the generic configuration class:

- `src/ggpeps/system/config_Z2.py`
- `Z2System2D_Config`

This config replaces the old G4-specific manager path and supports `ncopy=1` and even `ncopy` values at the level of the parameterization. The manager now constructs Z2 systems by passing `ncopy` explicitly from the command line.

The generic config has been smoke-tested for `ncopy=1`, `ncopy=2`, and `ncopy=4`. The `ncopy=8` path is currently not practical with the generic electric-energy projector construction: config construction becomes very expensive inside `generate_gauged_projector_terms()` before either exact evaluation or Monte Carlo evaluation begins.

#### Parameter-order compatibility

The generic Z2 config uses a systematic parameter order generated from `ncopy`. Saved parameter vectors from older config classes should not be assumed to be index-compatible with `Z2System2D_Config`.

When comparing against older results or loading old saved parameters, translate parameters by name/order rather than assuming that raw array indices match.

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

This refactor should be validated by comparing observables between the current branch and the latest `dev` branch for equivalent commands and equivalent parameter vectors. For configs whose parameter order changed, parameters must be translated by name before comparing results.

Detailed validation runs are tracked externally.