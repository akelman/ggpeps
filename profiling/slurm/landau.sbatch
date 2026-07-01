#!/bin/bash -l
#SBATCH --job-name=d6prof
#SBATCH --account=alex-account
#SBATCH --partition=alexq
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#
# Landau (HUJI) profiling job for the D6 MC study. CPU-only; handles BOTH the
# numpy-CPU and jax-CPU platforms via the BACKEND env var.
#
# Driven by exported env vars (see profiling/submit_matrix.sh). Required:
#   REPO VENV OUTDIR BACKEND(numpy|jax) RUN_KIND(phase|manager) TAG
#   SYS L LAYERS NCOPY GRADS(0|1) WARMUP MEAS
#   manager runs also: MODE(eval-mc|min-mc) NRUNNER [MAXITER METHOD]
#   optional deep-dive: DEEPDIVE(1) -> cProfile (numpy)

set -euo pipefail
: "${REPO:?}"; : "${VENV:?}"; : "${OUTDIR:?}"; : "${BACKEND:?}"; : "${RUN_KIND:?}"; : "${TAG:?}"
: "${SYS:=d6}"; : "${L:?}"; : "${LAYERS:=1}"; : "${NCOPY:=2}"; : "${GRADS:=1}"
: "${WARMUP:=500}"; : "${MEAS:=5000}"; : "${MODE:=eval-mc}"; : "${NRUNNER:=0}"
: "${MAXITER:=5}"; : "${METHOD:=bfgs}"; : "${DEEPDIVE:=0}"

source "$REPO/profiling/lib.sh"
source "$VENV"                       # activate venv (path to .../bin/activate)
cd "$REPO"

NCPU="${SLURM_CPUS_PER_TASK:-8}"
# thread budget: for Ray runners, split cores across runners; else use all.
if [[ "$NRUNNER" -gt 0 ]]; then
    PER=$(( NCPU / NRUNNER )); [[ "$PER" -lt 1 ]] && PER=1
else
    PER="$NCPU"
fi
if [[ "$BACKEND" == "jax" ]]; then set_jax_cpu_env "$PER"; else set_numpy_cpu_env "$PER"; fi

mkdir -p "$OUTDIR"
print_ledger | tee "$OUTDIR/ledger_${TAG}.txt"

GRADFLAG=""; [[ "$GRADS" == "1" ]] && GRADFLAG="--grads"
MGRGRAD="";  [[ "$GRADS" == "1" ]] && MGRGRAD="--compute_grads"

if [[ "$RUN_KIND" == "phase" ]]; then
    CP=""; [[ "$DEEPDIVE" == "1" && "$BACKEND" == "numpy" ]] && CP="--cprofile"
    $TIME_BIN python profiling/profile_mc_phases.py \
        --system "$SYS" --L "$L" --layers "$LAYERS" --ncopy "$NCOPY" $GRADFLAG \
        --warmup "$WARMUP" --meas "$MEAS" --tag "$TAG" \
        --out "$OUTDIR/phases.jsonl" $CP
else
    EXTRA=""; [[ "$MODE" == "min-mc" ]] && EXTRA="--maxiter $MAXITER --method $METHOD"
    $TIME_BIN python manager.py "$MODE" "${SYS^^}" --L "$L" --g 1.0 \
        --num_pg_layer "$LAYERS" --ncopy "$NCOPY" --nrunner "$NRUNNER" \
        --warmup_steps "$WARMUP" --meas_steps "$MEAS" $MGRGRAD $EXTRA \
        --ignore_cache --output "$OUTDIR"
fi
echo "DONE tag=$TAG"
