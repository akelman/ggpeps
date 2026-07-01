# D6 Monte-Carlo profiling harness

Profiles where D6 (and Z2) Monte-Carlo time goes, and how it scales with lattice
size L, across three platforms: **JAX-GPU (ALICE)**, **NumPy-CPU (Landau)**,
**JAX-CPU (Landau)**; axes: gradients on/off, Ray runners on/off, `eval-mc` vs
`min-mc`. All cluster outputs go to the user's personal `data/Dn/profiling/`.

## Pieces

| File | Role |
|---|---|
| `profile_mc_phases.py` | single-process phase + hotspot profiler (no `src/` edits — wraps hot methods at runtime). Backend chosen by `GGPEPS_BACKEND`. |
| `lib.sh` | shared slurm helpers: thread-fairness env, resource ledger, `nvidia-smi dmon` sampler. |
| `slurm/landau.sbatch` | CPU job (numpy **or** jax via `BACKEND`). |
| `slurm/alice_gpu.sbatch` | JAX-GPU job on the dedicated `gpu_lion` A100. |
| `slurm/alice_gpu_concurrency.sbatch` | launches **K** processes sharing the one A100 → max-jobs-per-GPU sweep (± MPS). |
| `submit_matrix.sh` | enumerates + `sbatch`es the focused matrix for a cluster. |
| `collect.py` | parse artifacts → tidy CSVs (run locally after pulling results). |

## Method notes (do not "fix" these)

- **cProfile only for NumPy.** It massively inflates JAX's eager dispatch. JAX uses
  `jax.profiler` traces + `JAX_LOG_COMPILES` + internal timers instead.
- **Thread fairness.** Both CPU backends get the same core+thread budget
  (`OMP/MKL/OPENBLAS` *and* XLA `intra_op_parallelism_threads`). Pinning only one
  backend produces fake ties.
- **Single A100 can't be split across jobs.** Multi-process GPU work = one sbatch,
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, small `MEM_FRACTION`.
- `config_build` (electric-term precompute) is a one-time cost — reported separately
  from per-step timing.

## Workflow

```bash
# on each cluster: get latest code + this harness
cd <repo> && git checkout Dn && git pull

# submit (paths inside the script — verify first)
bash profiling/submit_matrix.sh landau
bash profiling/submit_matrix.sh alice

# pull data/Dn/profiling/<date>/ back locally, then
python profiling/collect.py --base ./pulled_profiling
```

## Cleanup / revert

Profiling adds **no** changes to `src/` — everything lives under `profiling/`.
To revert: delete the `profiling/` directory. Personal-folder outputs under
`data/Dn/profiling/` can be removed independently.
