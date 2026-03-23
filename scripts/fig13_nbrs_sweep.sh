#!/bin/bash
set -e

# Figure 12: R_max (full_adj_nbrs) sweep — BigANN-100M
# nbrs = 0, 8, 16, 32, 48, 64 at T=1 and T=32
#
# Usage: ./scripts/fig13_nbrs_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${GATEANN_DATA_DIR:-${REPO_ROOT}/data}"
INDEX_DIR="${GATEANN_INDEX_DIR:-${DATA_DIR}/pipeann_indices}"
FILTER_DIR="${DATA_DIR}/filter_exp_100M"
RESULTS_DIR="${REPO_ROOT}/data/filter/results"
SEARCH_BIN="${REPO_ROOT}/build/tests/search_disk_index_fa"

mkdir -p "$RESULTS_DIR"

L_VALUES="20 30 40 50 70 100 150 200 250 300 400 500 700 1000 1500 2000"
NBRS_VALUES="0 8 16 32 48 64"

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

OUTFILE="${RESULTS_DIR}/fig_nbrs_sweep.txt"

{
echo "========================================"
echo " Figure 12: R_max (nbrs) Sweep (BigANN-100M, sel=10%)"
echo " $(date)"
echo "========================================"

for NBRS in $NBRS_VALUES; do
    for T in 1 32; do
        echo ""
        echo "[REPORT] GateANN(mode=8,nbrs=${NBRS}) sel=10% T=${T} bigann100M"
        log "GateANN nbrs=${NBRS} T=${T}..."
        for L in $L_VALUES; do
            run_search "$DTYPE" "$INDEX" "$T" 32 \
                "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
                8 10 0 $NBRS $L
        done
    done
done
} > "$OUTFILE"

log "R_max sweep done -> $OUTFILE"
log "=== fig13_nbrs_sweep.sh COMPLETE ==="
