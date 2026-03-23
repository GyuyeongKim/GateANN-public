#!/bin/bash
set -e

# Figure 5: Thread scaling — BigANN-100M
# DiskANN vs PipeANN vs GateANN at L=200, sel=10%, T=1..64
#
# Usage: ./scripts/fig06_thread_scaling.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${GATEANN_DATA_DIR:-${REPO_ROOT}/data}"
INDEX_DIR="${GATEANN_INDEX_DIR:-${DATA_DIR}/pipeann_indices}"
FILTER_DIR="${DATA_DIR}/filter_exp_100M"
RESULTS_DIR="${REPO_ROOT}/data/filter/results"
SEARCH_BIN="${REPO_ROOT}/build/tests/search_disk_index_fa"

mkdir -p "$RESULTS_DIR"

L=200
THREAD_VALUES="1 2 4 8 16 32 64"

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
# BigANN-100M
###############################################################################
DTYPE="uint8"
INDEX="${INDEX_DIR}/bigann100M"
QUERY="${DATA_DIR}/bigann100M_query.u8bin"
NLABELS="${FILTER_DIR}/bigann100M_node_labels.bin"
QLABELS="${FILTER_DIR}/bigann100M_query_labels.bin"
GT="${FILTER_DIR}/bigann100M_filtered_gt.bin"

OUTFILE="${RESULTS_DIR}/fig_thread_scaling.txt"

{
echo "========================================"
echo " Figure 5: Thread Scaling (BigANN-100M, L=${L}, sel=10%)"
echo " $(date)"
echo "========================================"

for T in $THREAD_VALUES; do
    # DiskANN: mode=0, BW=8, MEM_L=0
    echo ""
    echo "[REPORT] DiskANN(mode=0) sel=10% T=${T} bigann100M"
    log "DiskANN T=${T}..."
    run_search "$DTYPE" "$INDEX" "$T" 8 \
        "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
        0 0 0 $L
done

for T in $THREAD_VALUES; do
    # PipeANN: mode=2, BW=32, MEM_L=10
    echo ""
    echo "[REPORT] PipeANN(mode=2) sel=10% T=${T} bigann100M"
    log "PipeANN T=${T}..."
    run_search "$DTYPE" "$INDEX" "$T" 32 \
        "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
        2 10 0 $L
done

for T in $THREAD_VALUES; do
    # GateANN: mode=8, BW=32, MEM_L=10, full_adj_nbrs=32
    echo ""
    echo "[REPORT] GateANN(mode=8) sel=10% T=${T} bigann100M"
    log "GateANN T=${T}..."
    run_search "$DTYPE" "$INDEX" "$T" 32 \
        "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
        8 10 0 32 $L
done
} > "$OUTFILE"

log "Thread scaling done -> $OUTFILE"
log "=== fig06_thread_scaling.sh COMPLETE ==="
