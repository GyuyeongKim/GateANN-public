#!/bin/bash
set -e

# Figure 16: Pipeline depth (BW) sweep — BigANN-100M
# GateANN mode=8, varying BW=1,2,4,8,16,32,64
# T=1 and T=32, several L values
#
# Usage: ./scripts/fig16_bw_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${GATEANN_DATA_DIR:-${REPO_ROOT}/data}"
INDEX_DIR="${GATEANN_INDEX_DIR:-${DATA_DIR}/pipeann_indices}"
FILTER_DIR="${DATA_DIR}/filter_exp_100M"
RESULTS_DIR="${REPO_ROOT}/data/filter/results"
SEARCH_BIN="${REPO_ROOT}/build/tests/search_disk_index_fa"

mkdir -p "$RESULTS_DIR"

L_VALUES="20 30 40 50 70 100 150 200 300 400 500 700 1000"
BW_VALUES="1 2 4 8 16 32 64"

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

OUTFILE="${RESULTS_DIR}/fig_bw_sweep.txt"

{
echo "========================================"
echo " Figure 16: Pipeline Depth (BW) Sweep (BigANN-100M, sel=10%)"
echo " $(date)"
echo "========================================"

for BW in $BW_VALUES; do
    for T in 1 32; do
        # GateANN: mode=8, MEM_L=10, full_adj_nbrs=32
        # Note: BW=1 effectively disables pipelining (synchronous)
        echo ""
        echo "[REPORT] GateANN(mode=8,BW=${BW}) sel=10% T=${T} bigann100M"
        log "GateANN BW=${BW} T=${T}..."
        for L in $L_VALUES; do
            run_search "$DTYPE" "$INDEX" "$T" "$BW" \
                "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
                8 10 0 32 $L
        done
    done
done
} > "$OUTFILE"

log "BW sweep done -> $OUTFILE"
log "=== fig16_bw_sweep.sh COMPLETE ==="
