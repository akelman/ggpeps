#!/bin/bash -l
#SBATCH --job-name=d6prof_gpuK
#SBATCH --account=gpu_lion
#SBATCH --partition=gpu_lion
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#
# GPU-CONCURRENCY SWEEP on the single gpu_lion A100.
# Launches K identical D6 MC processes that SHARE the one GPU, then measures
# per-process ms/step and aggregate throughput to find how many jobs usefully
# fit on one A100 in this partition. Optionally routes them through an
# nvidia-cuda-mps-control daemon (MPS) to test small-kernel concurrency.
#
# Required env: REPO VENV OUTDIR K L ; optional: SYS LAYERS NCOPY GRADS WARMUP MEAS
#   USE_MPS(0|1) MEMFRAC(default ~1/K)

set -euo pipefail
: "${REPO:?}"; : "${VENV:?}"; : "${OUTDIR:?}"; : "${K:?}"; : "${L:?}"
: "${SYS:=d6}"; : "${LAYERS:=1}"; : "${NCOPY:=2}"; : "${GRADS:=1}"
: "${WARMUP:=300}"; : "${MEAS:=3000}"; : "${USE_MPS:=0}"

source "$REPO/profiling/lib.sh"
source "$VENV"
cd "$REPO"

# default per-process GPU memory fraction ~ 1/K with headroom, floored.
if [[ -z "${MEMFRAC:-}" ]]; then
    MEMFRAC=$(python -c "print(max(0.05, min(0.9, 0.9/float($K))))")
fi
set_jax_gpu_env "$MEMFRAC"
# each proc gets a slice of host cores
NCPU="${SLURM_CPUS_PER_TASK:-16}"
PER=$(( NCPU / K )); [[ "$PER" -lt 1 ]] && PER=1
set_cpu_threads "$PER"

MPSTAG="nomps"
if [[ "$USE_MPS" == "1" ]]; then
    MPSTAG="mps"
    export CUDA_MPS_PIPE_DIRECTORY="$OUTDIR/mps_pipe_${SLURM_JOB_ID}"
    export CUDA_MPS_LOG_DIRECTORY="$OUTDIR/mps_log_${SLURM_JOB_ID}"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d || echo "[warn] MPS daemon failed to start"
fi

RUNTAG="K${K}_${MPSTAG}_L${L}"
mkdir -p "$OUTDIR"
print_ledger | tee "$OUTDIR/ledger_${RUNTAG}.txt"
echo "K=$K MEMFRAC=$MEMFRAC USE_MPS=$USE_MPS PER_CPU=$PER" | tee -a "$OUTDIR/ledger_${RUNTAG}.txt"

start_dmon "$OUTDIR/dmon_${RUNTAG}.log"
trap 'stop_dmon; [[ "$USE_MPS" == "1" ]] && echo quit | nvidia-cuda-mps-control 2>/dev/null || true' EXIT

GRADFLAG=""; [[ "$GRADS" == "1" ]] && GRADFLAG="--grads"

t0=$(date +%s.%N)
pids=()
for i in $(seq 1 "$K"); do
    python profiling/profile_mc_phases.py \
        --system "$SYS" --L "$L" --layers "$LAYERS" --ncopy "$NCOPY" $GRADFLAG \
        --warmup "$WARMUP" --meas "$MEAS" --seed $(( 1000 + i )) \
        --tag "${RUNTAG}_p${i}" --out "$OUTDIR/concurrency.jsonl" \
        > "$OUTDIR/proc_${RUNTAG}_p${i}.out" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
t1=$(date +%s.%N)

stop_dmon
WALL=$(python -c "print(round($t1-$t0,3))")
echo "CONCURRENCY_SWEEP tag=$RUNTAG K=$K wall_all_procs_sec=$WALL" | tee -a "$OUTDIR/ledger_${RUNTAG}.txt"
echo "DONE $RUNTAG"
