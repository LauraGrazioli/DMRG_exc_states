#!/bin/bash
# One irrep of tda_water_julia.jl at M = 500 on a 32-core node (the recorded runs: Owl, AMD EPYC 9454).
#   sbatch --export=ALL,IRREP=A1 run_tda_owl.slurm        (A1 ~2.4 h, B1/A2/B2 ~3.2-3.8 h at 32 threads)
# Fill in your account and partition. Run from water_nakatani_tda/; QNTensorTrains must be a sibling checkout of
# DMRG_exc_states (or set QNTENSORTRAINS_PATH), instantiated once with
#   julia --project=<QNTensorTrains> -e 'using Pkg; Pkg.instantiate()'.
#SBATCH -A <account>
#SBATCH -p <partition>
#SBATCH -N 1 --ntasks=1 --cpus-per-task=32 --mem=240G
#SBATCH -t 12:00:00
#SBATCH -o slurm-%j.out
set -euo pipefail
M=${M:-500}
mkdir -p logs states
export OPENBLAS_NUM_THREADS=1 SAVE_DIR=states POLISH_GTOL=${POLISH_GTOL:-1e-8}
{
  echo "=== $IRREP M=$M | node $(hostname) | $(lscpu | sed -n 's/^Model name: *//p') | job ${SLURM_JOB_ID:-} | $(date -Is)"
  julia -t $SLURM_CPUS_PER_TASK tda_water_julia.jl $M $IRREP
  echo "=== done $(date -Is)"
} > logs/julia_m${M}_$IRREP.log 2>&1
