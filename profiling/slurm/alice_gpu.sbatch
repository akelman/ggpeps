#!/bin/bash -l
#SBATCH --job-name=d6prof_gpu
#SBATCH --account=gpu_lion
#SBATCH --partition=gpu_lion
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#
# ALICE (Leiden) JAX-GPU profiling job on the dedicated gpu_lion A100.
# Handles the single-process phase runs and the manager.py end-to-end / nrunner
# runs. The single A100 cannot be split across jobs -> nrunner runs share this
# one GPU inside one job (XLA_PYTHON_CLIENT_PREALLOCATE=false).
#
# Required env: REPO VENV OUTDIR RUN_KIND(phase|manager) TAG SYS L LAYERS NCOPY
#   GRADS WARMUP MEAS ; manager also MODE NRUNNER [MAXITER METHOD] ; DEEPDIVE(1)->jax-trace

set -euo pipefail
: "${REPO:?}"; : "${VENV:?}"; : "${OUTDIR:?}"; : "${RUN_KIND:?}"; : "${TAG:?}"
: "${SYS:=d6}"; : "${L:?}"; : "${LAYERS:=1}"; : "${NCOPY:=2}"; : "${GRADS:=1}"
: "${WARMUP:=500}"; : "${MEAS:=5000}"; : "${MODE:=eval-mc}"; : "${NRUNNER:=0}"
: "${MAXITER:=5}"; : "${METHOD:=bfgs}"; : "${DEEPDIVE:=0}"; : "${MEMFRAC:=0.4}"

source "$REPO/profiling/lib.sh"
source "$VENV"
cd "$REPO"

set_jax_gpu_env "$MEMFRAC"
NCPU="${SLURM_CPUS_PER_TASK:-8}"
set_cpu_threads "$NCPU"              # host-side BLAS budget too

mkdir -p "$OUTDIR"
print_ledger | tee "$OUTDIR/ledger_${TAG}.txt"

# sample GPU utilization/memory during the run
start_dmon "$OUTDIR/dmon_${TAG}.log"
trap stop_dmon EXIT

GRADFLAG=""; [[ "$GRADS" == "1" ]] && GRADFLAG="--grads"
MGRGRAD="";  [[ "$GRADS" == "1" ]] && MGRGRAD="--compute_grads"

if [[ "$RUN_KIND" == "phase" ]]; then
    TR=""; [[ "$DEEPDIVE" == "1" ]] && TR="--jax-trace $OUTDIR/jaxtrace_${TAG}"
    $TIME_BIN python profiling/profile_mc_phases.py \
        --system "$SYS" --L "$L" --layers "$LAYERS" --ncopy "$NCOPY" $GRADFLAG \
        --warmup "$WARMUP" --meas "$MEAS" --tag "$TAG" \
        --out "$OUTDIR/phases.jsonl" $TR
else
    EXTRA=""; [[ "$MODE" == "min-mc" ]] && EXTRA="--maxiter $MAXITER --method $METHOD"
    PJ=""; [[ "$DEEPDIVE" == "1" ]] && PJ="--profile_jax"
    $TIME_BIN python manager.py "$MODE" "${SYS^^}" --L "$L" --g 1.0 \
        --num_pg_layer "$LAYERS" --ncopy "$NCOPY" --nrunner "$NRUNNER" \
        --warmup_steps "$WARMUP" --meas_steps "$MEAS" $MGRGRAD $EXTRA $PJ \
        --ignore_cache --output "$OUTDIR"
fi
stop_dmon
echo "DONE tag=$TAG"
