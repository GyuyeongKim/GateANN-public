#!/bin/bash
set -e

# Master script: run all per-figure experiments, then generate all figures.
#
# Usage: ./scripts/run_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

TOTAL_START=$(date +%s)

###############################################################################
# Phase 1: Run all experiment scripts sequentially
###############################################################################
EXPERIMENT_SCRIPTS=(
    "$SCRIPT_DIR/fig01_motivation.sh"
    "$SCRIPT_DIR/fig04_pareto_main.sh"
    "$SCRIPT_DIR/fig05_thread_scaling.sh"
    "$SCRIPT_DIR/fig06_io_reduction.sh"
    "$SCRIPT_DIR/fig07_billion.sh"
    "$SCRIPT_DIR/fig08_yfcc.sh"
    "$SCRIPT_DIR/fig09_vamana.sh"
    "$SCRIPT_DIR/fig10_fdiskann.sh"
    "$SCRIPT_DIR/fig11_selectivity.sh"
    "$SCRIPT_DIR/fig12_nbrs_sweep.sh"
    "$SCRIPT_DIR/fig13_zipf.sh"
    "$SCRIPT_DIR/fig14_spatial.sh"
    "$SCRIPT_DIR/fig15_range.sh"
    "$SCRIPT_DIR/fig16_bw_sweep.sh"
    "$SCRIPT_DIR/fig17_ablation.sh"
)

log "=== Phase 1: Running ${#EXPERIMENT_SCRIPTS[@]} experiment scripts ==="
echo ""

FAILED=()
for script in "${EXPERIMENT_SCRIPTS[@]}"; do
    name=$(basename "$script")
    if [ ! -f "$script" ]; then
        log "SKIP $name (not found)"
        continue
    fi

    log "START $name"
    START=$(date +%s)

    if bash "$script"; then
        ELAPSED=$(( $(date +%s) - START ))
        log "DONE  $name (${ELAPSED}s)"
    else
        ELAPSED=$(( $(date +%s) - START ))
        log "FAIL  $name (${ELAPSED}s)"
        FAILED+=("$name")
    fi
    echo ""
done

###############################################################################
# Phase 2: Generate all figures
###############################################################################
log "=== Phase 2: Generating figures ==="
cd "$REPO_ROOT"
python3 scripts/generate_all_figures.py

###############################################################################
# Summary
###############################################################################
TOTAL_ELAPSED=$(( $(date +%s) - TOTAL_START ))
echo ""
log "=== ALL DONE (total ${TOTAL_ELAPSED}s) ==="

if [ ${#FAILED[@]} -gt 0 ]; then
    log "WARNING: ${#FAILED[@]} script(s) failed: ${FAILED[*]}"
    exit 1
fi
