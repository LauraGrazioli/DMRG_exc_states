#!/bin/bash
# Runs tda_water_rs.jl (QNTensorTrains-rs >= 0.1.1) one irrep at a time and writes
# logs/rs_m<M>_<irrep>.log, then the Table IV summary of those logs (summarize_tda.jl).
#   ./run_tda_rs.sh                  all four irreps at M = 500, one after the other
#   ./run_tda_rs.sh 500 A1 B2        chosen irreps
# Environment: QNTT_THREADS (library threads, default all cores), POLISH_GTOL (1e-8), TDA_TOL (1e-6),
# QNTT_RS_PATH (default ../../QNTensorTrains-rs), QNTT_LIB (the library file, if not in the checkout's
# target directory). Each irrep saves its polished reference as states/<irrep>_m<M>.tt and reuses it.
set -euo pipefail
cd "$(dirname "$0")"
M=${1:-500}; shift || true
IRREPS=${*:-A1 B1 A2 B2}
mkdir -p logs states
export SAVE_DIR=states POLISH_GTOL=${POLISH_GTOL:-1e-8}
logs=()
for IRREP in $IRREPS; do
  log=logs/rs_m${M}_$IRREP.log
  logs+=("$log")
  echo "$IRREP: $log"
  {
    echo "=== $IRREP M=$M | $(hostname) | $(date)"
    julia tda_water_rs.jl "$M" "$IRREP"
    echo "=== done $(date)"
  } > "$log" 2>&1
done
julia summarize_tda.jl "${logs[@]}"
