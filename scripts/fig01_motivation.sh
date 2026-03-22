#!/bin/bash
set -e

# Figure 1: Motivation — BigANN-100M, sel=10%
# (a) Thread scaling: DiskANN + PipeANN, T=1..32, L=200
# (b) Naive pre-filter collapse: mode=8 nbrs=0 (no tunneling) vs mode=2 (post-filter)
#
# Usage: ./scripts/fig01_motivation.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${GATEANN_DATA_DIR:-${REPO_ROOT}/data}"
INDEX_DIR="${GATEANN_INDEX_DIR:-${DATA_DIR}/pipeann_indices}"
FILTER_DIR="${DATA_DIR}/filter_exp_100M"
RESULTS_DIR="${REPO_ROOT}/data/filter/results"
SEARCH_BIN="${REPO_ROOT}/build/tests/search_disk_index_fa"

mkdir -p "$RESULTS_DIR"

L=200
L_VALUES="20 30 40 50 70 100 150 200 250 300 400 500 700 1000 1500 2000"
THREAD_VALUES="1 2 4 8 16 32"

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

OUTFILE="${RESULTS_DIR}/fig_motivation.txt"

{
echo "========================================"
echo " Figure 1: Motivation (BigANN-100M, sel=10%)"
echo " $(date)"
echo "========================================"

###############################################################################
# Part (a): Thread scaling — DiskANN + PipeANN at L=200
###############################################################################
for T in $THREAD_VALUES; do
    echo ""
    echo "[REPORT] DiskANN(mode=0) sel=10% T=${T} bigann100M"
    log "DiskANN T=${T} L=${L}..."
    run_search "$DTYPE" "$INDEX" "$T" 8 \
        "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
        0 0 0 $L
done

for T in $THREAD_VALUES; do
    echo ""
    echo "[REPORT] PipeANN(mode=2) sel=10% T=${T} bigann100M"
    log "PipeANN T=${T} L=${L}..."
    run_search "$DTYPE" "$INDEX" "$T" 32 \
        "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
        2 10 0 $L
done

###############################################################################
# Part (b): Naive pre-filter collapse — nbrs=0 (no tunneling) vs post-filter
###############################################################################
for T in 1 32; do
    # Naive pre-filter: mode=8, nbrs=0 => filter check but NO FullAdjIndex tunneling
    echo ""
    echo "[REPORT] NaivePreFilter(mode=8,nbrs=0) sel=10% T=${T} bigann100M"
    log "NaivePreFilter T=${T}..."
    for L in $L_VALUES; do
        run_search "$DTYPE" "$INDEX" "$T" 32 \
            "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
            8 10 0 0 $L
    done

    # Post-filter baseline: mode=2
    echo ""
    echo "[REPORT] PipeANN(mode=2) sel=10% T=${T} bigann100M"
    log "PipeANN (post-filter) T=${T}..."
    for L in $L_VALUES; do
        run_search "$DTYPE" "$INDEX" "$T" 32 \
            "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
            2 10 0 $L
    done
done
} > "$OUTFILE"

log "Motivation done -> $OUTFILE"
log "=== fig01_motivation.sh COMPLETE ==="
