#!/bin/bash
set -e

# Figure 5: Main Pareto curves — BigANN-100M + DEEP-100M
# DiskANN vs PipeANN vs GateANN, T=1 and T=32
#
# Usage: ./scripts/fig05_pareto_main.sh

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
    local mode_args="$@"   # mode mem_L cache_budget [full_adj_nbrs] L...

    "$SEARCH_BIN" "$dtype" "$index" "$threads" "$bw" \
        "$query" "$nlabels" "$qlabels" "$gt" \
        10 l2 pq $mode_args 2>&1 | grep -P '^\s+\d+\s' || true
}

###############################################################################
# BigANN-100M
###############################################################################
BIGANN_DTYPE="uint8"
BIGANN_INDEX="${INDEX_DIR}/bigann100M"
BIGANN_QUERY="${DATA_DIR}/bigann100M_query.u8bin"
BIGANN_NLABELS="${FILTER_DIR}/bigann100M_node_labels.bin"
BIGANN_QLABELS="${FILTER_DIR}/bigann100M_query_labels.bin"
BIGANN_GT="${FILTER_DIR}/bigann100M_filtered_gt.bin"

BIGANN_OUT="${RESULTS_DIR}/fig_pareto_bigann100M_main.txt"

{
echo "========================================"
echo " Figure 5: BigANN-100M Pareto (sel=10%)"
echo " $(date)"
echo "========================================"

for T in 1 32; do
    # DiskANN: mode=0, BW=8, MEM_L=0
    echo ""
    echo "[REPORT] DiskANN(mode=0) sel=10% T=${T} bigann100M"
    log "BigANN DiskANN T=${T}..."
    for L in $L_VALUES; do
        run_search "$BIGANN_DTYPE" "$BIGANN_INDEX" "$T" 8 \
            "$BIGANN_QUERY" "$BIGANN_NLABELS" "$BIGANN_QLABELS" "$BIGANN_GT" \
            0 0 0 $L
    done

    # PipeANN: mode=2, BW=32, MEM_L=10
    echo ""
    echo "[REPORT] PipeANN(mode=2) sel=10% T=${T} bigann100M"
    log "BigANN PipeANN T=${T}..."
    for L in $L_VALUES; do
        run_search "$BIGANN_DTYPE" "$BIGANN_INDEX" "$T" 32 \
            "$BIGANN_QUERY" "$BIGANN_NLABELS" "$BIGANN_QLABELS" "$BIGANN_GT" \
            2 10 0 $L
    done

    # GateANN: mode=8, BW=32, MEM_L=10, full_adj_nbrs=32
    echo ""
    echo "[REPORT] GateANN(mode=8) sel=10% T=${T} bigann100M"
    log "BigANN GateANN T=${T}..."
    for L in $L_VALUES; do
        run_search "$BIGANN_DTYPE" "$BIGANN_INDEX" "$T" 32 \
            "$BIGANN_QUERY" "$BIGANN_NLABELS" "$BIGANN_QLABELS" "$BIGANN_GT" \
            8 10 0 32 $L
    done
done
} > "$BIGANN_OUT"

log "BigANN-100M Pareto done -> $BIGANN_OUT"

###############################################################################
# DEEP-100M
###############################################################################
DEEP_DTYPE="float"
DEEP_INDEX="${INDEX_DIR}/deep100M"
DEEP_QUERY="${DATA_DIR}/deep100M_query.fbin"
DEEP_NLABELS="${FILTER_DIR}/deep100M_node_labels.bin"
DEEP_QLABELS="${FILTER_DIR}/deep100M_query_labels.bin"
DEEP_GT="${FILTER_DIR}/deep100M_filtered_gt.bin"

DEEP_OUT="${RESULTS_DIR}/fig_pareto_deep100M_main.txt"

{
echo "========================================"
echo " Figure 5: DEEP-100M Pareto (sel=10%)"
echo " $(date)"
echo "========================================"

for T in 1 32; do
    # DiskANN
    echo ""
    echo "[REPORT] DiskANN(mode=0) sel=10% T=${T} deep100M"
    log "DEEP DiskANN T=${T}..."
    for L in $L_VALUES; do
        run_search "$DEEP_DTYPE" "$DEEP_INDEX" "$T" 8 \
            "$DEEP_QUERY" "$DEEP_NLABELS" "$DEEP_QLABELS" "$DEEP_GT" \
            0 0 0 $L
    done

    # PipeANN
    echo ""
    echo "[REPORT] PipeANN(mode=2) sel=10% T=${T} deep100M"
    log "DEEP PipeANN T=${T}..."
    for L in $L_VALUES; do
        run_search "$DEEP_DTYPE" "$DEEP_INDEX" "$T" 32 \
            "$DEEP_QUERY" "$DEEP_NLABELS" "$DEEP_QLABELS" "$DEEP_GT" \
            2 10 0 $L
    done

    # GateANN
    echo ""
    echo "[REPORT] GateANN(mode=8) sel=10% T=${T} deep100M"
    log "DEEP GateANN T=${T}..."
    for L in $L_VALUES; do
        run_search "$DEEP_DTYPE" "$DEEP_INDEX" "$T" 32 \
            "$DEEP_QUERY" "$DEEP_NLABELS" "$DEEP_QLABELS" "$DEEP_GT" \
            8 10 0 32 $L
    done
done
} > "$DEEP_OUT"

log "DEEP-100M Pareto done -> $DEEP_OUT"
log "=== fig05_pareto_main.sh COMPLETE ==="
