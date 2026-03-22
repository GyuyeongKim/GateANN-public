#!/bin/bash
set -e

# Figure 6: I/O reduction analysis — BigANN-100M
# GateANN mode=8, T=1, varying L and selectivity
#
# Usage: ./scripts/fig06_io_reduction.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${GATEANN_DATA_DIR:-${REPO_ROOT}/data}"
INDEX_DIR="${GATEANN_INDEX_DIR:-${DATA_DIR}/pipeann_indices}"
FILTER_DIR="${DATA_DIR}/filter_exp_100M"
RESULTS_DIR="${REPO_ROOT}/data/filter/results"
SEARCH_BIN="${REPO_ROOT}/build/tests/search_disk_index_fa"

mkdir -p "$RESULTS_DIR"

L_VALUES="20 30 40 50 70 100 150 200 250 300 400 500 700 1000 1500 2000"
T=1

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

OUTFILE="${RESULTS_DIR}/fig_io_reduction.txt"

{
echo "========================================"
echo " Figure 6: I/O Reduction (BigANN-100M, T=${T})"
echo " $(date)"
echo "========================================"

# Part A: Varying L at sel=10% — PipeANN (baseline IOs) vs GateANN (reduced IOs)
NLABELS="${FILTER_DIR}/bigann100M_node_labels.bin"
QLABELS="${FILTER_DIR}/bigann100M_query_labels.bin"
GT="${FILTER_DIR}/bigann100M_filtered_gt.bin"

echo ""
echo "[REPORT] PipeANN(mode=2) sel=10% T=${T} bigann100M"
log "PipeANN baseline (sel=10%, T=${T})..."
for L in $L_VALUES; do
    run_search "$DTYPE" "$INDEX" "$T" 32 \
        "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
        2 10 0 $L
done

echo ""
echo "[REPORT] GateANN(mode=8) sel=10% T=${T} bigann100M"
log "GateANN (sel=10%, T=${T})..."
for L in $L_VALUES; do
    run_search "$DTYPE" "$INDEX" "$T" 32 \
        "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
        8 10 0 32 $L
done

# Part B: Varying selectivity (5%, 10%, 20%) at full L sweep
for SEL in 5 10 20; do
    if [ "$SEL" -eq 10 ]; then
        NL="${FILTER_DIR}/bigann100M_node_labels.bin"
        QL="${FILTER_DIR}/bigann100M_query_labels.bin"
        G="${FILTER_DIR}/bigann100M_filtered_gt.bin"
    else
        NL="${FILTER_DIR}/bigann100M_sel${SEL}pct_node_labels.bin"
        QL="${FILTER_DIR}/bigann100M_sel${SEL}pct_query_labels.bin"
        G="${FILTER_DIR}/bigann100M_sel${SEL}pct_filtered_gt.bin"
    fi

    echo ""
    echo "[REPORT] PipeANN(mode=2) sel=${SEL}% T=${T} bigann100M"
    log "PipeANN baseline (sel=${SEL}%, T=${T})..."
    for L in $L_VALUES; do
        run_search "$DTYPE" "$INDEX" "$T" 32 \
            "$QUERY" "$NL" "$QL" "$G" \
            2 10 0 $L
    done

    echo ""
    echo "[REPORT] GateANN(mode=8) sel=${SEL}% T=${T} bigann100M"
    log "GateANN (sel=${SEL}%, T=${T})..."
    for L in $L_VALUES; do
        run_search "$DTYPE" "$INDEX" "$T" 32 \
            "$QUERY" "$NL" "$QL" "$G" \
            8 10 0 32 $L
    done
done
} > "$OUTFILE"

log "I/O reduction done -> $OUTFILE"
log "=== fig06_io_reduction.sh COMPLETE ==="
