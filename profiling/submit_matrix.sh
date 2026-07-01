#!/bin/bash
# Enumerate + submit the FOCUSED D6 MC profiling matrix on a given cluster.
#
#   bash profiling/submit_matrix.sh landau     # -> numpy-CPU + jax-CPU jobs
#   bash profiling/submit_matrix.sh alice      # -> jax-GPU jobs + concurrency sweep
#
# Cluster-specific paths (REPO/VENV/OUTDIR) are set below; verify before running.
# All outputs land in the user's PERSONAL data folder under data/Dn/profiling.

set -euo pipefail
CLUSTER="${1:?usage: submit_matrix.sh <landau|alice>}"
STAMP="$(date +%Y%m%d)"

case "$CLUSTER" in
  landau)
    REPO="/home/data/itayg/code/gaussian-peps"     # VERIFY on cluster
    VENV="/home/data/itayg/pyenv/gaussian_peps/bin/activate"
    OUTROOT="/home/data/itayg/data/Dn/profiling/$STAMP"
    SBATCH="$REPO/profiling/slurm/landau.sbatch"
    BACKENDS="numpy jax"
    MEMG_DEFAULT=12
    ;;
  alice)
    REPO="/data1/projects/pi-emontsp/itayg/code/gaussian-peps"
    VENV="/data1/projects/pi-emontsp/pyenvs/ggpeps_new/bin/activate"
    OUTROOT="/data1/projects/pi-emontsp/itayg/data/Dn/profiling/$STAMP"
    SBATCH="$REPO/profiling/slurm/alice_gpu.sbatch"
    BACKENDS="gpu"
    MEMG_DEFAULT=64
    ;;
  *) echo "unknown cluster $CLUSTER"; exit 1;;
esac

mkdir -p "$OUTROOT"
echo "cluster=$CLUSTER repo=$REPO out=$OUTROOT"

Ls="2 4 6"
WARMUP=500; MEAS=5000
FIXED_CPU=8; MEMG="${MEMG_DEFAULT:-12}" # 8 CPUs @ 12G schedules immediately on alexq
NCPU_SWEEP="1 2 4 8 16"                 # CPU-scaling sweep -> optimal cpus per L
SWEEP_WARMUP=200; SWEEP_MEAS=2000       # shorter (many jobs; still averages plenty)

submit() {  # submit <sbatch> <tag> <cpus> <extra EXPORTS...>
    local script="$1"; local tag="$2"; local cpus="$3"; shift 3
    local exports="REPO=$REPO,VENV=$VENV,OUTDIR=$OUTROOT,TAG=$tag,SYS=d6,LAYERS=1,NCOPY=2,$*"
    echo "sbatch -> $tag (cpus=$cpus)"
    sbatch --export="ALL,$exports" --job-name="$tag" \
           --cpus-per-task="$cpus" --mem="${MEMG}G" \
           -o "$OUTROOT/slurm_%x_%j.out" -e "$OUTROOT/slurm_%x_%j.out" "$script"
}

for L in $Ls; do
  # ===== CPU-SCALING SWEEP (landau only): optimal #cpus per L, per backend =====
  # single-process phase harness, grads on; thread budget auto-tracks cpus.
  if [[ "$CLUSTER" == landau ]]; then
    for B in $BACKENDS; do
      for C in $NCPU_SWEEP; do
        submit "$SBATCH" "cpusweep_${B}_L${L}_c${C}" "$C" \
          "BACKEND=$B,RUN_KIND=phase,L=$L,GRADS=1,WARMUP=$SWEEP_WARMUP,MEAS=$SWEEP_MEAS"
      done
    done
  fi

  # ----- A. eval-mc: grad x {0,1}, nrunner x {0,4}, phase(single-proc) + manager(end-to-end)
  for GRADS in 0 1; do
    # phase-level hotspot breakdown (single process) at the fixed cpu budget
    if [[ "$CLUSTER" == landau ]]; then
      for B in $BACKENDS; do
        submit "$SBATCH" "phase_${B}_L${L}_g${GRADS}" "$FIXED_CPU" \
          "BACKEND=$B,RUN_KIND=phase,L=$L,GRADS=$GRADS,WARMUP=$WARMUP,MEAS=$MEAS"
      done
    else
      submit "$SBATCH" "phase_gpu_L${L}_g${GRADS}" "$FIXED_CPU" \
        "RUN_KIND=phase,L=$L,GRADS=$GRADS,WARMUP=$WARMUP,MEAS=$MEAS"
    fi
    # end-to-end wall + nrunner comparison (manager.py)
    for NR in 0 4; do
      if [[ "$CLUSTER" == landau ]]; then
        for B in $BACKENDS; do
          submit "$SBATCH" "evalmc_${B}_L${L}_g${GRADS}_nr${NR}" "$FIXED_CPU" \
            "BACKEND=$B,RUN_KIND=manager,MODE=eval-mc,L=$L,GRADS=$GRADS,NRUNNER=$NR,WARMUP=$WARMUP,MEAS=$MEAS"
        done
      else
        submit "$SBATCH" "evalmc_gpu_L${L}_g${GRADS}_nr${NR}" "$FIXED_CPU" \
          "RUN_KIND=manager,MODE=eval-mc,L=$L,GRADS=$GRADS,NRUNNER=$NR,WARMUP=$WARMUP,MEAS=$MEAS"
      fi
    done
  done

  # ----- B. min-mc carryover: grads always on, nrunner x {0,4}, short
  for NR in 0 4; do
    if [[ "$CLUSTER" == landau ]]; then
      for B in $BACKENDS; do
        submit "$SBATCH" "minmc_${B}_L${L}_nr${NR}" "$FIXED_CPU" \
          "BACKEND=$B,RUN_KIND=manager,MODE=min-mc,L=$L,GRADS=1,NRUNNER=$NR,WARMUP=1000,MEAS=5000,MAXITER=5"
      done
    else
      submit "$SBATCH" "minmc_gpu_L${L}_nr${NR}" "$FIXED_CPU" \
        "RUN_KIND=manager,MODE=min-mc,L=$L,GRADS=1,NRUNNER=$NR,WARMUP=1000,MEAS=5000,MAXITER=5"
    fi
  done
done

# ----- C. GPU concurrency sweep (ALICE only) -----
if [[ "$CLUSTER" == alice ]]; then
  CONC="$REPO/profiling/slurm/alice_gpu_concurrency.sbatch"
  for L in 4 6; do
    for K in 1 2 4 8 16; do
      for MPS in 0 1; do
        tag="conc_K${K}_mps${MPS}_L${L}"
        exports="REPO=$REPO,VENV=$VENV,OUTDIR=$OUTROOT,K=$K,L=$L,USE_MPS=$MPS,SYS=d6,LAYERS=1,NCOPY=2,GRADS=1"
        echo "sbatch -> $tag"
        sbatch --export="ALL,$exports" --job-name="$tag" \
               -o "$OUTROOT/slurm_%x_%j.out" -e "$OUTROOT/slurm_%x_%j.out" "$CONC"
      done
    done
  done
fi

echo "submitted matrix for $CLUSTER"
