#!/bin/bash
set -e

# Figure 14: Spatial label correlation — BigANN-100M
# k-means correlated labels at alpha=0.0, 0.25, 0.5, 0.75, 1.0
# GateANN only, T=32
#
# Usage: ./scripts/fig14_spatial.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${GATEANN_DATA_DIR:-${REPO_ROOT}/data}"
INDEX_DIR="${GATEANN_INDEX_DIR:-${DATA_DIR}/pipeann_indices}"
SPATIAL_DIR="${DATA_DIR}/filter_exp_100M/spatial"
RESULTS_DIR="${REPO_ROOT}/data/filter/results"
SEARCH_BIN="${REPO_ROOT}/build/tests/search_disk_index_fa"

mkdir -p "$RESULTS_DIR"

L_VALUES="20 30 40 50 70 100 150 200 250 300 400 500 700 1000"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

###############################################################################
# Helpers
###############################################################################
run_search() {
    local dtype=$1 index=$2 threads=$3 bw=$4 query=$5 nlabels=$6 qlabels=$7 gt=$8
    shift 8
    local mode_args="$@"

    "$SEARCH_BIN" "$dtype" "$index" "$threads" "$bw" \
        "$query" "$nlabels" "$qlabels" "$gt" \
        10 l2 pq $mode_args 2>&1 | grep -P '^\s+\d+\s' || true
}

###############################################################################
# BigANN-100M — Spatial correlation sweep
###############################################################################
DTYPE="uint8"
INDEX="${INDEX_DIR}/bigann100M"
QUERY="${DATA_DIR}/bigann100M_query.u8bin"

OUTFILE="${RESULTS_DIR}/fig_spatial.txt"

{
echo "========================================"
echo " Figure 14: Spatial Label Correlation (BigANN-100M, T=32)"
echo " alpha: 0.0 (random) -> 1.0 (fully correlated)"
echo " $(date)"
echo "========================================"

for ALPHA in 0.00 0.25 0.50 0.75 1.00; do
    # Derive filename tag: alpha_0.00 -> alpha_0_00
    ALPHA_TAG=$(echo "$ALPHA" | sed 's/\./_/')

    NLABELS="${SPATIAL_DIR}/bigann100M_node_labels_alpha_${ALPHA}.bin"
    QLABELS="${SPATIAL_DIR}/bigann100M_query_labels_alpha_${ALPHA}.bin"
    GT="${SPATIAL_DIR}/bigann100M_filtered_gt_alpha_${ALPHA}.bin"

    # PipeANN (mode=2), T=32 — baseline for comparison
    echo ""
    echo "[REPORT] PipeANN(mode=2) alpha=${ALPHA} T=32 bigann100M"
    log "PipeANN alpha=${ALPHA} T=32..."
    for L in $L_VALUES; do
        run_search "$DTYPE" "$INDEX" 32 32 \
            "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
            2 10 0 $L
    done

    # GateANN (mode=8, nbrs=32), T=32
    echo ""
    echo "[REPORT] GateANN(mode=8) alpha=${ALPHA} T=32 bigann100M"
    log "GateANN alpha=${ALPHA} T=32..."
    for L in $L_VALUES; do
        run_search "$DTYPE" "$INDEX" 32 32 \
            "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
            8 10 0 32 $L
    done

    log "  alpha=${ALPHA} done."
done
} > "$OUTFILE"

log "Spatial correlation done -> $OUTFILE"
log "=== fig14_spatial.sh COMPLETE ==="
