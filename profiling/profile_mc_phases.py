"""Single-process D6/Z2 Monte-Carlo phase + hotspot profiler.

Backend-aware profiling harness used for the D6 MC performance study across
NumPy-CPU, JAX-CPU and JAX-GPU. It builds the requested ansatz, runs a short MC
evaluation, and records:

  * coarse PHASE timings  : warmup (eager loop), run(update), run(measure),
                            grad-aggregation, and precompute/init.
  * per-FUNCTION hotspots : a curated set of hot-path methods wrapped at runtime
                            (no edits to src/ needed) with call-count + total time.
  * a RESOURCE ledger     : backend, cgroup CPU budget, thread env, GPU info,
                            git hash, and all run parameters.

Everything is emitted as ONE json object (also appended as a json-line to --out).

Profiling-method rules baked in (see memory / plan):
  * cProfile MASSIVELY inflates JAX -> only enabled for the numpy backend
    (via --cprofile), and run as a SEPARATE pass so it never contaminates the
    clean phase timers.
  * JAX timings are honest because measure()/calculate_weight_attempt() call
    float()/np.asarray(), forcing device sync; we additionally block on warmup.
  * Never pin only one backend's threads -- the slurm wrappers set OMP/MKL/XLA
    thread budgets equally; this script just records what it was given.

Usage (backend is chosen by GGPEPS_BACKEND in the environment, BEFORE import):
    GGPEPS_BACKEND=numpy python profiling/profile_mc_phases.py \
        --system d6 --L 4 --grads --warmup 500 --meas 5000 \
        --out /home/data/itayg/data/Dn/profiling/results.jsonl --tag landau_numpy

    GGPEPS_BACKEND=jax python profiling/profile_mc_phases.py \
        --system d6 --L 6 --grads --warmup 500 --meas 5000 \
        --jax-trace /path/to/trace_dir --out results.jsonl --tag alice_gpu
"""
import argparse
import json
import os
import platform
import socket
import subprocess
import time
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# runtime function-timer: wraps bound methods to accumulate (calls, seconds)
# ---------------------------------------------------------------------------
class FuncTimers:
    """Accumulate wall time + call counts for a set of wrapped callables."""

    def __init__(self):
        self.calls = defaultdict(int)
        self.secs = defaultdict(float)
        self._orig = {}  # (obj, name) -> original attr

    def wrap_method(self, obj, name, label=None):
        """Monkeypatch obj.name so every call is timed. Idempotent-safe."""
        label = label or name
        if not hasattr(obj, name):
            return False
        orig = getattr(obj, name)
        self._orig[(id(obj), name)] = (obj, name, orig)

        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return orig(*args, **kwargs)
            finally:
                self.secs[label] += time.perf_counter() - t0
                self.calls[label] += 1

        setattr(obj, name, wrapper)
        return True

    def restore(self):
        for obj, name, orig in self._orig.values():
            setattr(obj, name, orig)
        self._orig.clear()

    def as_dict(self):
        return {
            k: {"calls": self.calls[k], "secs": round(self.secs[k], 6),
                "ms_per_call": round(1e3 * self.secs[k] / max(self.calls[k], 1), 6)}
            for k in sorted(self.secs, key=lambda x: -self.secs[x])
        }


# ---------------------------------------------------------------------------
# system construction (mirrors timing_tracker.py)
# ---------------------------------------------------------------------------
def make_system(system_name, L, layers, ncopy, seed):
    from ggpeps import lattice, system

    lat = lattice.Lattice2D(L, L, 0)  # no gauge fixing
    rng = np.random.default_rng(seed)
    if system_name == "z2":
        cfg = system.Z2System2D_G2C_F2C_Config(
            lat, 1.0, 1.0, 0.0, 0.0, None, num_pg_layer=layers, num_fermionic_layer=0, mod_link_inds=(0,)
        )
        sys_cls = system.Z2System2D
    elif system_name == "d6":
        cfg = system.D6System2D_Config(
            lat, 1.0, 1.0, 0.0, 0.0, None, num_pg_layer=layers, num_fermionic_layer=0, mod_link_inds=(0,)
        )
        sys_cls = system.D2nSystem2D
    else:
        raise ValueError(system_name)

    cfg.paramvec = rng.standard_normal(cfg.param_shape())
    if isinstance(cfg, system.Z2System2DConfig):
        cfg.make_pure_gauge()
    cfg.enforce_parameter_conditions(cfg.paramvec)
    return cfg, sys_cls


# ---------------------------------------------------------------------------
# resource ledger
# ---------------------------------------------------------------------------
def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,uuid", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out.splitlines()
    except Exception:
        return []


def _cpu_budget():
    """Best-effort CPUs available to this process (cgroup/affinity aware)."""
    try:
        n_aff = len(os.sched_getaffinity(0))  # honours cpuset/affinity (Linux)
    except AttributeError:
        n_aff = os.cpu_count()
    return n_aff


def resource_ledger(args, backend):
    env_keys = [
        "GGPEPS_BACKEND", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE",
        "SLURM_JOB_PARTITION", "SLURM_JOB_ACCOUNT",
    ]
    return {
        "hostname": socket.gethostname(),
        "backend": backend,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "n_cpu_total": os.cpu_count(),
        "n_cpu_available": _cpu_budget(),
        "gpu": _gpu_info(),
        "git_hash": _git_hash(),
        "env": {k: os.environ.get(k) for k in env_keys if os.environ.get(k) is not None},
    }


# ---------------------------------------------------------------------------
# core profiling run
# ---------------------------------------------------------------------------
def profile_run(args):
    import ggpeps
    from ggpeps.mc import MonteCarloEvaluator, MonteCarloEvaluatorConfig

    backend = getattr(ggpeps, "PREFERRED_BACKEND", os.environ.get("GGPEPS_BACKEND", "numpy"))
    is_jax = backend == "jax"

    result = {
        "tag": args.tag,
        "system": args.system,
        "L": args.L,
        "layers": args.layers,
        "ncopy": args.ncopy,
        "grads": bool(args.grads),
        "warmup_steps": args.warmup,
        "meas_steps": args.meas,
        "update_size": args.update_size,
        "seed": args.seed,
        "resource": resource_ledger(args, backend),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    timers = FuncTimers()

    # --- build + init (precompute phase) ---
    t0 = time.perf_counter()
    cfg, sys_cls = make_system(args.system, args.L, args.layers, args.ncopy, args.seed)
    t_cfg = time.perf_counter() - t0

    t0 = time.perf_counter()
    sysobj = sys_cls(cfg)
    sysobj.initialize()
    t_init = time.perf_counter() - t0

    mc_cfg = MonteCarloEvaluatorConfig(
        warmup_steps=args.warmup, meas_steps=args.meas, compute_grads=bool(args.grads),
        update_size_per_step=args.update_size,
    )
    mc_cfg.seed = args.seed
    ev = MonteCarloEvaluator(mc_cfg, sysobj)

    # --- wrap hot-path methods on the system + evaluator ---
    for name in ["calculate_weight_attempt", "update_gauge_ind", "generate_rotmat",
                 "calculate_update_gamma_in", "update_lognorm_inc", "calculate_lognorm"]:
        timers.wrap_method(sysobj, name)
    timers.wrap_method(ev, "update", label="ev.update")
    timers.wrap_method(ev, "measure", label="ev.measure")

    # --- warmup phase (eager loop) ---
    t0 = time.perf_counter()
    ev.warmup()
    t_warmup = time.perf_counter() - t0

    # --- run phase (update + measure). ev.update/ev.measure are timed separately
    #     via the wrappers, so t_run splits into those two accumulators. ---
    t0 = time.perf_counter()
    ev.run()
    t_run = time.perf_counter() - t0

    # grad aggregation happens inside ev.run() (energy_gradient_mc) already; time
    # a standalone call too so we can size it explicitly.
    t_gradagg = 0.0
    if args.grads:
        t0 = time.perf_counter()
        _ = ev.energy_gradient_mc()
        t_gradagg = time.perf_counter() - t0

    fn = timers.as_dict()
    timers.restore()

    total_steps = args.warmup + args.meas
    acc = ev.obsdict["acceptance_prob"].mean() if len(ev.obsdict["acceptance_prob"]) else None

    result["phases_sec"] = {
        "config_build": round(t_cfg, 6),
        "system_init_precompute": round(t_init, 6),
        "warmup_eager_loop": round(t_warmup, 6),
        "run_total": round(t_run, 6),
        "run_update": fn.get("ev.update", {}).get("secs", 0.0),
        "run_measure": fn.get("ev.measure", {}).get("secs", 0.0),
        "grad_aggregation_standalone": round(t_gradagg, 6),
    }
    result["derived"] = {
        "ms_per_warmup_step": round(1e3 * t_warmup / max(args.warmup, 1), 6),
        "ms_per_run_step": round(1e3 * t_run / max(args.meas, 1), 6),
        "ms_per_step_overall": round(1e3 * (t_warmup + t_run) / max(total_steps, 1), 6),
        "acceptance": None if acc is None else round(float(acc), 4),
    }
    result["functions"] = fn
    return result, ev


def run_cprofile(args, out_dir):
    """NumPy-only: cProfile a short evaluate() and dump top functions."""
    import cProfile
    import io
    import pstats

    import ggpeps
    from ggpeps.mc import MonteCarloEvaluator, MonteCarloEvaluatorConfig

    if getattr(ggpeps, "PREFERRED_BACKEND", "numpy") == "jax":
        return None  # never cProfile jax

    cfg, sys_cls = make_system(args.system, args.L, args.layers, args.ncopy, args.seed)
    sysobj = sys_cls(cfg)
    sysobj.initialize()
    mc_cfg = MonteCarloEvaluatorConfig(
        warmup_steps=min(args.warmup, 200), meas_steps=min(args.meas, 2000),
        compute_grads=bool(args.grads), update_size_per_step=args.update_size,
    )
    mc_cfg.seed = args.seed
    ev = MonteCarloEvaluator(mc_cfg, sysobj)

    pr = cProfile.Profile()
    pr.enable()
    ev.evaluate()
    pr.disable()

    os.makedirs(out_dir, exist_ok=True)
    stats_path = os.path.join(out_dir, f"cprofile_{args.tag}_{args.system}_L{args.L}.pstats")
    pr.dump_stats(stats_path)

    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(35)
    txt_path = os.path.join(out_dir, f"cprofile_{args.tag}_{args.system}_L{args.L}.txt")
    with open(txt_path, "w") as f:
        f.write(s.getvalue())
    return stats_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=["z2", "d6"], default="d6")
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--ncopy", type=int, default=2)
    ap.add_argument("--grads", action="store_true", default=False)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--meas", type=int, default=5000)
    ap.add_argument("--update_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--out", type=str, default="profiling_results.jsonl",
                    help="json-lines file to append the result to")
    ap.add_argument("--cprofile", action="store_true", default=False,
                    help="also run a numpy-only cProfile pass (ignored on jax)")
    ap.add_argument("--jax-trace", type=str, default=None,
                    help="if set and backend==jax, capture a jax.profiler trace to this dir")
    args = ap.parse_args()

    # optional jax device trace around the whole clean run
    trace_ctx = None
    if args.jax_trace:
        try:
            import ggpeps
            if getattr(ggpeps, "PREFERRED_BACKEND", "numpy") == "jax":
                import jax
                os.makedirs(args.jax_trace, exist_ok=True)
                jax.profiler.start_trace(args.jax_trace)
                trace_ctx = jax
        except Exception as e:  # pragma: no cover
            print(f"[warn] could not start jax trace: {e}")

    result, _ev = profile_run(args)

    if trace_ctx is not None:
        trace_ctx.profiler.stop_trace()
        result["jax_trace_dir"] = args.jax_trace

    if args.cprofile:
        out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
        stats_path = run_cprofile(args, out_dir)
        if stats_path:
            result["cprofile"] = stats_path

    # emit
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
