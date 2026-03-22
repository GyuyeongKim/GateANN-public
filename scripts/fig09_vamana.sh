#!/bin/bash
set -e

# Figure 9: In-memory Vamana comparison — BigANN-100M
# Vamana (search_mem_fa, in-memory upper bound) vs GateANN (search_disk_index_fa)
# T=1 and T=32
#
# Usage: ./scripts/fig09_vamana.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${GATEANN_DATA_DIR:-${REPO_ROOT}/data}"
INDEX_DIR="${GATEANN_INDEX_DIR:-${DATA_DIR}/pipeann_indices}"
FILTER_DIR="${DATA_DIR}/filter_exp_100M"
RESULTS_DIR="${REPO_ROOT}/data/filter/results"
SEARCH_DISK_BIN="${REPO_ROOT}/build/tests/search_disk_index_fa"
SEARCH_MEM_BIN="${REPO_ROOT}/build/tests/search_mem_fa"

mkdir -p "$RESULTS_DIR"

L_VALUES="20 30 40 50 70 100 150 200 250 300 400 500 700 1000 1500 2000"

# search_mem_fa arg format:
#   <type> <index_prefix> <num_threads> <query_bin>
#   <node_labels_bin> <query_labels_bin> <filtered_gt_bin>
#   <K> <dist_metric>
#   <L1> [L2] ...

log() { echo "[$(date '+%H:%M:%S')] $*"; }

###############################################################################
# Helpers
###############################################################################
run_disk_search() {
    local dtype=$1 index=$2 threads=$3 bw=$4 query=$5 nlabels=$6 qlabels=$7 gt=$8
    shift 8
    local mode_args="$@"

    "$SEARCH_DISK_BIN" "$dtype" "$index" "$threads" "$bw" \
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

OUTFILE="${RESULTS_DIR}/fig_vamana.txt"

{
echo "========================================"
echo " Figure 9: In-Memory Vamana vs GateANN (BigANN-100M, sel=10%)"
echo " $(date)"
echo "========================================"

for T in 1 32; do
    # Vamana (in-memory, post-filter) — ideal upper bound
    echo ""
    echo "[REPORT] Vamana(in-memory) sel=10% T=${T} bigann100M"
    log "Vamana T=${T}..."
    "$SEARCH_MEM_BIN" "$DTYPE" "$INDEX" "$T" \
        "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
        10 l2 $L_VALUES 2>&1 | grep -P '^\s+\d+\s' || true

    # GateANN: mode=8, BW=32, MEM_L=10, full_adj_nbrs=32
    echo ""
    echo "[REPORT] GateANN(mode=8) sel=10% T=${T} bigann100M"
    log "GateANN T=${T}..."
    for L in $L_VALUES; do
        run_disk_search "$DTYPE" "$INDEX" "$T" 32 \
            "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
            8 10 0 32 $L
    done
done
} > "$OUTFILE"

log "Vamana comparison done -> $OUTFILE"
log "=== fig09_vamana.sh COMPLETE ==="
