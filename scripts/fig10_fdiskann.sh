#!/bin/bash
set -e

# Figure 10: Filtered-DiskANN comparison — BigANN-100M
# Filtered-DiskANN (mode=10) vs GateANN (mode=8), T=1 and T=32
# mode=10 uses synchronous beam_search with hard candidate filter + per-label medoids
#
# Usage: ./scripts/fig10_fdiskann.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${GATEANN_DATA_DIR:-${REPO_ROOT}/data}"
INDEX_DIR="${GATEANN_INDEX_DIR:-${DATA_DIR}/pipeann_indices}"
FILTER_DIR="${DATA_DIR}/filter_exp_100M"
RESULTS_DIR="${REPO_ROOT}/data/filter/results"
SEARCH_BIN="${REPO_ROOT}/build/tests/search_disk_index_fa"

mkdir -p "$RESULTS_DIR"

L_VALUES="20 30 40 50 70 100 150 200 250 300 400 500 700 1000 1500 2000"

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

OUTFILE="${RESULTS_DIR}/fig_fdiskann.txt"

{
echo "========================================"
echo " Figure 10: Filtered-DiskANN vs GateANN (BigANN-100M, sel=10%)"
echo " $(date)"
echo "========================================"

for T in 1 32; do
    # Filtered-DiskANN: mode=10, BW=8 (synchronous beam_search)
    echo ""
    echo "[REPORT] FilteredDiskANN(mode=10) sel=10% T=${T} bigann100M"
    log "Filtered-DiskANN T=${T}..."
    for L in $L_VALUES; do
        run_search "$DTYPE" "$INDEX" "$T" 8 \
            "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
            10 0 0 $L
    done

    # DiskANN: mode=0, BW=8 (post-filter baseline for reference)
    echo ""
    echo "[REPORT] DiskANN(mode=0) sel=10% T=${T} bigann100M"
    log "DiskANN T=${T}..."
    for L in $L_VALUES; do
        run_search "$DTYPE" "$INDEX" "$T" 8 \
            "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
            0 0 0 $L
    done

    # PipeANN: mode=2, BW=32
    echo ""
    echo "[REPORT] PipeANN(mode=2) sel=10% T=${T} bigann100M"
    log "PipeANN T=${T}..."
    for L in $L_VALUES; do
        run_search "$DTYPE" "$INDEX" "$T" 32 \
            "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
            2 10 0 $L
    done

    # GateANN: mode=8, BW=32, MEM_L=10, full_adj_nbrs=32
    echo ""
    echo "[REPORT] GateANN(mode=8) sel=10% T=${T} bigann100M"
    log "GateANN T=${T}..."
    for L in $L_VALUES; do
        run_search "$DTYPE" "$INDEX" "$T" 32 \
            "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
            8 10 0 32 $L
    done
done
} > "$OUTFILE"

log "Filtered-DiskANN comparison done -> $OUTFILE"
log "=== fig10_fdiskann.sh COMPLETE ==="
