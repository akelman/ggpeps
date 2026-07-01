#!/bin/bash
# Shared helpers for the D6 MC profiling slurm jobs.
# Sourced by the per-platform sbatch scripts. Keeps the resource ledger and the
# thread-fairness controls in one place (see plan / memory: never pin only one
# backend's threads).

set -euo pipefail

# ---- thread budgets --------------------------------------------------------
# Give BOTH cpu backends the same core budget so the comparison is fair.
# n = number of threads (usually = cpus-per-task, or cpus/nrunner for runners).
set_cpu_threads() {
    local n="$1"
    export OMP_NUM_THREADS="$n"
    export MKL_NUM_THREADS="$n"
    export OPENBLAS_NUM_THREADS="$n"
    export NUMEXPR_NUM_THREADS="$n"
}

# JAX-CPU also needs its XLA threadpool sized, else it oversubscribes / undersubscribes.
set_jax_cpu_env() {
    local n="$1"
    set_cpu_threads "$n"
    export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${n}"
    export GGPEPS_BACKEND=jax
}

set_numpy_cpu_env() {
    local n="$1"
    set_cpu_threads "$n"
    export GGPEPS_BACKEND=numpy
}

# JAX-GPU: do NOT preallocate the whole A100, so K processes can share the one GPU.
# mem_fraction is passed in (~1/K) for the concurrency sweep; default modest.
set_jax_gpu_env() {
    local memfrac="${1:-0.15}"
    export GGPEPS_BACKEND=jax
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION="$memfrac"
}

# ---- resource ledger -------------------------------------------------------
print_ledger() {
    echo "==================== RESOURCE LEDGER ===================="
    echo "date            : $(date -Is)"
    echo "hostname        : $(hostname)"
    echo "slurm job id    : ${SLURM_JOB_ID:-none}"
    echo "slurm partition : ${SLURM_JOB_PARTITION:-none}"
    echo "slurm account   : ${SLURM_JOB_ACCOUNT:-none}"
    echo "cpus-per-task   : ${SLURM_CPUS_PER_TASK:-unset}"
    echo "mem-per-node    : ${SLURM_MEM_PER_NODE:-unset}"
    echo "cpu affinity    : $(python -c 'import os;print(len(os.sched_getaffinity(0)))' 2>/dev/null || echo '?')"
    echo "GGPEPS_BACKEND  : ${GGPEPS_BACKEND:-unset}"
    echo "OMP_NUM_THREADS : ${OMP_NUM_THREADS:-unset}"
    echo "XLA_FLAGS       : ${XLA_FLAGS:-unset}"
    echo "XLA_PY_PREALLOC : ${XLA_PYTHON_CLIENT_PREALLOCATE:-unset}"
    echo "XLA_PY_MEMFRAC  : ${XLA_PYTHON_CLIENT_MEM_FRACTION:-unset}"
    echo "CUDA_VISIBLE    : ${CUDA_VISIBLE_DEVICES:-unset}"
    echo "git HEAD        : $(git -C "${REPO:-.}" rev-parse --short HEAD 2>/dev/null || echo '?')"
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "--- nvidia-smi ---"
        nvidia-smi --query-gpu=name,memory.total,uuid --format=csv,noheader || true
    fi
    echo "========================================================"
}

# Prefer GNU time -v (max RSS etc.) if available, else fall back to plain run.
TIME_BIN=""
if command -v /usr/bin/time >/dev/null 2>&1; then
    TIME_BIN="/usr/bin/time -v"
fi

# start nvidia-smi dmon sampler in background; returns its PID via $DMON_PID.
start_dmon() {
    local logf="$1"
    DMON_PID=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi dmon -s um -d 1 -o DT > "$logf" 2>/dev/null &
        DMON_PID=$!
    fi
}
stop_dmon() {
    if [[ -n "${DMON_PID:-}" ]]; then
        kill "$DMON_PID" 2>/dev/null || true
    fi
}
