#!/bin/bash
# EXPERIMENTAL. Runs QNTensorTrains' GAD driver (examples/gad_molecule.jl: gentlest-ascent dynamics for an index-1
# saddle on the fixed-rank SU(2) manifold) on this directory's water cc-pVDZ 8e/23o FCIDUMP, one irrep at a time, and
# writes logs/gad_m<M>_<irrep>.log. Per irrep: an in-irrep reference at cap M (aufbau CSF -> in-irrep MALS -> typed
# CG), then GAD PINNED to the irrep (typed manifold; targets the irrep's second singlet), then GAD UNPINNED with the
# symmetry monitor (off-irrep weights of the state, the gradient and the tracked mode, per step).
#   ./run_gad.sh                  all four irreps at M = 25, one after the other
#   ./run_gad.sh 25 A1 B2         chosen irreps
# Environment: THREADS (default: all cores), QNTENSORTRAINS_PATH (default ../../QNTensorTrains), MAX_STEPS (1000 GAD
# steps per run), GAD_V0 (1 | 0: seed the tracked mode with the kick direction [default] or with the lowest term-A
# mode, the Rust convention), GAD_STEP (newton [default]: Rust's step controller + a Newton tail on the reflected
# Hessian once ||g|| <= 1e-2 | rust: Euler steps only, as in the Rust code), GAD_RUNS (both [default] | pinned).
# First use: julia --project=$QNTENSORTRAINS_PATH/examples -e 'using Pkg; Pkg.instantiate()'
#
# Measured at M = 25 (8 threads, Apple Silicon, two runs side by side), pinned runs, GAD_V0=1, GAD_STEP=newton vs
# rust: every irrep converges to a certified in-irrep index-1 saddle, the same saddle under both step rules -- A2 81
# steps / 70 s (rust: 479 / 295 s), B1 661 / 590 s (rust: hit the 1000-step cap at ||g|| 6.6e-7), B2 331 / 363 s
# (rust: 951 / 1362 s), A1 218 / 316 s (rust: 545 / 329 s). With GAD_V0=0, A1 undid the kick and was refused as a
# border point. Unpinned (GAD_V0=0, rust): the tracked mode lies outside the irrep in every case and 3 of 4 runs
# hit the step budget. At M = 25 the saddles are 40-64 mEh above the FCI levels: results are rank-constrained
# saddles, not converged excitation energies.
set -euo pipefail
cd "$(dirname "$0")"
M=${1:-25}; shift || true
IRREPS=${*:-A1 B1 A2 B2}
THREADS=${THREADS:-$(julia -e 'print(Sys.CPU_THREADS)')}
QNT=${QNTENSORTRAINS_PATH:-../../QNTensorTrains}
mkdir -p logs
export OPENBLAS_NUM_THREADS=1 FCI_FILE="$PWD/fci_vdz.txt" GAD_V0=${GAD_V0:-1}
for IRREP in $IRREPS; do
  log=logs/gad_m${M}_$IRREP.log
  echo "$IRREP: $log"
  {
    echo "=== $IRREP M=$M | $(hostname) | threads $THREADS | GAD_V0=$GAD_V0 | GAD_STEP=${GAD_STEP:-newton} | $(date)"
    julia -t "$THREADS" --project="$QNT/examples" "$QNT/examples/gad_molecule.jl" "$PWD/FCIDUMP" "$M" "$IRREP" "${MAX_STEPS:-1000}"
    echo "=== done $(date)"
  } > "$log" 2>&1
done
