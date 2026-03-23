#!/bin/bash
set -e

# Figure 15: Range predicates — BigANN-100M
# L2-norm based bins (10 equal-frequency bins)
# DiskANN vs PipeANN vs GateANN, T=1 and T=32
#
# Usage: ./scripts/fig16_range.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${GATEANN_DATA_DIR:-${REPO_ROOT}/data}"
INDEX_DIR="${GATEANN_INDEX_DIR:-${DATA_DIR}/pipeann_indices}"
RANGE_DIR="${DATA_DIR}/filter_exp_100M/range"
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
# BigANN-100M — Range predicate (L2-norm bins)
###############################################################################
DTYPE="uint8"
INDEX="${INDEX_DIR}/bigann100M"
QUERY="${DATA_DIR}/bigann100M_query.u8bin"
NLABELS="${RANGE_DIR}/bigann100M_node_labels_range.bin"
QLABELS="${RANGE_DIR}/bigann100M_query_labels_range.bin"
GT="${RANGE_DIR}/bigann100M_filtered_gt_range.bin"

OUTFILE="${RESULTS_DIR}/fig_range.txt"

{
echo "========================================"
echo " Figure 15: Range Predicate (BigANN-100M, L2-norm bins)"
echo " $(date)"
echo "========================================"

for T in 1 32; do
    # DiskANN: mode=0, BW=8, MEM_L=0
    echo ""
    echo "[REPORT] DiskANN(mode=0) range T=${T} bigann100M"
    log "DiskANN T=${T} (range)..."
    for L in $L_VALUES; do
        run_search "$DTYPE" "$INDEX" "$T" 8 \
            "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
            0 0 0 $L
    done

    # PipeANN: mode=2, BW=32, MEM_L=10
    echo ""
    echo "[REPORT] PipeANN(mode=2) range T=${T} bigann100M"
    log "PipeANN T=${T} (range)..."
    for L in $L_VALUES; do
        run_search "$DTYPE" "$INDEX" "$T" 32 \
            "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
            2 10 0 $L
    done

    # GateANN: mode=8, BW=32, MEM_L=10, full_adj_nbrs=32
    echo ""
    echo "[REPORT] GateANN(mode=8) range T=${T} bigann100M"
    log "GateANN T=${T} (range)..."
    for L in $L_VALUES; do
        run_search "$DTYPE" "$INDEX" "$T" 32 \
            "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
            8 10 0 32 $L
    done
done
} > "$OUTFILE"

log "Range predicate done -> $OUTFILE"
log "=== fig16_range.sh COMPLETE ==="
