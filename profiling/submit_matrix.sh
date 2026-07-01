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
    ;;
  alice)
    REPO="/data1/projects/pi-emontsp/itayg/code/gaussian-peps"
    VENV="/data1/projects/pi-emontsp/pyenvs/ggpeps_new/bin/activate"
    OUTROOT="/data1/projects/pi-emontsp/itayg/data/Dn/profiling/$STAMP"
    SBATCH="$REPO/profiling/slurm/alice_gpu.sbatch"
    BACKENDS="gpu"
    ;;
  *) echo "unknown cluster $CLUSTER"; exit 1;;
esac

mkdir -p "$OUTROOT"
echo "cluster=$CLUSTER repo=$REPO out=$OUTROOT"

Ls="2 4 6"
WARMUP=500; MEAS=5000

submit() {  # submit <sbatch> <tag> <extra EXPORTS...>
    local script="$1"; local tag="$2"; shift 2
    local exports="REPO=$REPO,VENV=$VENV,OUTDIR=$OUTROOT,TAG=$tag,SYS=d6,LAYERS=1,NCOPY=2,$*"
    echo "sbatch --export=ALL,$exports --job-name=$tag $script"
    sbatch --export="ALL,$exports" --job-name="$tag" "$script"
}

for L in $Ls; do
  # ----- A. eval-mc: grad x {0,1}, nrunner x {0,4}, phase(single-proc) + manager(end-to-end)
  for GRADS in 0 1; do
    # phase-level hotspot breakdown (single process)
    if [[ "$CLUSTER" == landau ]]; then
      for B in $BACKENDS; do
        submit "$SBATCH" "phase_${B}_L${L}_g${GRADS}" \
          "BACKEND=$B,RUN_KIND=phase,L=$L,GRADS=$GRADS,WARMUP=$WARMUP,MEAS=$MEAS"
      done
    else
      submit "$SBATCH" "phase_gpu_L${L}_g${GRADS}" \
        "RUN_KIND=phase,L=$L,GRADS=$GRADS,WARMUP=$WARMUP,MEAS=$MEAS"
    fi
    # end-to-end wall + nrunner comparison (manager.py)
    for NR in 0 4; do
      if [[ "$CLUSTER" == landau ]]; then
        for B in $BACKENDS; do
          submit "$SBATCH" "evalmc_${B}_L${L}_g${GRADS}_nr${NR}" \
            "BACKEND=$B,RUN_KIND=manager,MODE=eval-mc,L=$L,GRADS=$GRADS,NRUNNER=$NR,WARMUP=$WARMUP,MEAS=$MEAS"
        done
      else
        submit "$SBATCH" "evalmc_gpu_L${L}_g${GRADS}_nr${NR}" \
          "RUN_KIND=manager,MODE=eval-mc,L=$L,GRADS=$GRADS,NRUNNER=$NR,WARMUP=$WARMUP,MEAS=$MEAS"
      fi
    done
  done

  # ----- B. min-mc carryover: grads always on, nrunner x {0,4}, short
  for NR in 0 4; do
    if [[ "$CLUSTER" == landau ]]; then
      for B in $BACKENDS; do
        submit "$SBATCH" "minmc_${B}_L${L}_nr${NR}" \
          "BACKEND=$B,RUN_KIND=manager,MODE=min-mc,L=$L,GRADS=1,NRUNNER=$NR,WARMUP=1000,MEAS=5000,MAXITER=5"
      done
    else
      submit "$SBATCH" "minmc_gpu_L${L}_nr${NR}" \
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
        echo "sbatch --export=ALL,$exports --job-name=$tag $CONC"
        sbatch --export="ALL,$exports" --job-name="$tag" "$CONC"
      done
    done
  done
fi

echo "submitted matrix for $CLUSTER"
