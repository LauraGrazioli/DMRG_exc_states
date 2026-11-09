#!/bin/bash
# Runs tda_water_julia.jl one irrep at a time and writes logs/julia_m<M>_<irrep>.log, the names summarize_tda.jl reads.
#   ./run_tda.sh                  all four irreps at M = 500, one after the other
#   ./run_tda.sh 500 A1 B2        chosen irreps
# Environment: THREADS (default: all cores), POLISH_GTOL (1e-8), QNTENSORTRAINS_PATH (default ../../QNTensorTrains).
# Each irrep saves its polished reference as states/<irrep>.jls.
#
# Recorded at M = 500 (32 threads, AMD EPYC 9454): 2.4 h (A1) to 3.8 h (A2) per irrep, 17-21 GB peak memory per irrep,
# about 2.5 cores busy on average. Run irreps in parallel only with ~24 GB of memory free for each.
set -euo pipefail
cd "$(dirname "$0")"
M=${1:-500}; shift || true
IRREPS=${*:-A1 B1 A2 B2}
THREADS=${THREADS:-$(julia -e 'print(Sys.CPU_THREADS)')}
mkdir -p logs states
export OPENBLAS_NUM_THREADS=1 SAVE_DIR=states POLISH_GTOL=${POLISH_GTOL:-1e-8}
for IRREP in $IRREPS; do
  log=logs/julia_m${M}_$IRREP.log
  echo "$IRREP: $log"
  {
    echo "=== $IRREP M=$M | $(hostname) | threads $THREADS | $(date)"
    julia -t "$THREADS" tda_water_julia.jl "$M" "$IRREP"
    echo "=== done $(date)"
  } > "$log" 2>&1
done
julia summarize_tda.jl
