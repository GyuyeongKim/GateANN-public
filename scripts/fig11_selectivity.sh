#!/bin/bash
set -e

# Figure 11: Selectivity sweep — BigANN-100M
# 5% / 10% / 20% selectivity, T=32, full L sweep
#
# Usage: ./scripts/fig11_selectivity.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${GATEANN_DATA_DIR:-${REPO_ROOT}/data}"
INDEX_DIR="${GATEANN_INDEX_DIR:-${DATA_DIR}/pipeann_indices}"
FILTER_DIR="${DATA_DIR}/filter_exp_100M"
RESULTS_DIR="${REPO_ROOT}/data/filter/results"
SEARCH_BIN="${REPO_ROOT}/build/tests/search_disk_index_fa"

mkdir -p "$RESULTS_DIR"

L_VALUES="20 30 40 50 70 100 150 200 250 300 400 500 700 1000 1500 2000"
T=32

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

OUTFILE="${RESULTS_DIR}/fig_selectivity.txt"

{
echo "========================================"
echo " Figure 11: Selectivity Sweep (BigANN-100M, T=${T})"
echo " $(date)"
echo "========================================"

for SEL in 5 10 20; do
    if [ "$SEL" -eq 10 ]; then
        NLABELS="${FILTER_DIR}/bigann100M_node_labels.bin"
        QLABELS="${FILTER_DIR}/bigann100M_query_labels.bin"
        GT="${FILTER_DIR}/bigann100M_filtered_gt.bin"
    else
        NLABELS="${FILTER_DIR}/bigann100M_sel${SEL}pct_node_labels.bin"
        QLABELS="${FILTER_DIR}/bigann100M_sel${SEL}pct_query_labels.bin"
        GT="${FILTER_DIR}/bigann100M_sel${SEL}pct_filtered_gt.bin"
    fi

    # PipeANN: mode=2, BW=32, MEM_L=10
    echo ""
    echo "[REPORT] PipeANN(mode=2) sel=${SEL}% T=${T} bigann100M"
    log "PipeANN sel=${SEL}% T=${T}..."
    for L in $L_VALUES; do
        run_search "$DTYPE" "$INDEX" "$T" 32 \
            "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
            2 10 0 $L
    done

    # GateANN: mode=8, BW=32, MEM_L=10, full_adj_nbrs=32
    echo ""
    echo "[REPORT] GateANN(mode=8) sel=${SEL}% T=${T} bigann100M"
    log "GateANN sel=${SEL}% T=${T}..."
    for L in $L_VALUES; do
        run_search "$DTYPE" "$INDEX" "$T" 32 \
            "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
            8 10 0 32 $L
    done
done
} > "$OUTFILE"

log "Selectivity sweep done -> $OUTFILE"
log "=== fig11_selectivity.sh COMPLETE ==="
