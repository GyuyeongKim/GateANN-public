#!/bin/bash
set -e

# Figure 7: BigANN-1B Pareto curves
# DiskANN vs PipeANN vs GateANN, T=1 and T=32
#
# Usage: ./scripts/fig08_billion.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${GATEANN_DATA_DIR:-${REPO_ROOT}/data}"
INDEX_DIR="${GATEANN_INDEX_DIR:-${DATA_DIR}/pipeann_indices}"
FILTER_DIR="${DATA_DIR}/filter_exp_1B"
RESULTS_DIR="${REPO_ROOT}/data/filter/results"
SEARCH_BIN="${REPO_ROOT}/build/tests/search_disk_index_fa"

mkdir -p "$RESULTS_DIR"

L_VALUES="20 30 40 50 70 100 150 200 300 400 500 700 1000"

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
# BigANN-1B
###############################################################################
DTYPE="uint8"
INDEX="${INDEX_DIR}/bigann1B"
QUERY="${DATA_DIR}/bigann1B_query.u8bin"
NLABELS="${FILTER_DIR}/bigann1B_node_labels.bin"
QLABELS="${FILTER_DIR}/bigann1B_query_labels.bin"
GT="${FILTER_DIR}/bigann1B_filtered_gt.bin"

# Verify required files
for f in "${INDEX}_disk.index" "${INDEX}_pq_compressed.bin" \
         "$NLABELS" "$QLABELS" "$GT"; do
    if [ ! -f "$f" ]; then
        echo "MISSING: $f"
        echo "Run setup_data.sh bigann1B first, then build the index."
        exit 1
    fi
done

OUTFILE="${RESULTS_DIR}/fig_pareto_bigann1B_main.txt"

{
echo "========================================"
echo " Figure 7: BigANN-1B Pareto (sel=10%)"
echo " $(date)"
echo "========================================"

for T in 1 32; do
    # DiskANN: mode=0, BW=8, MEM_L=0
    echo ""
    echo "[REPORT] DiskANN(mode=0) sel=10% T=${T} bigann1B"
    log "BigANN-1B DiskANN T=${T}..."
    for L in $L_VALUES; do
        run_search "$DTYPE" "$INDEX" "$T" 8 \
            "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
            0 0 0 $L
    done

    # PipeANN: mode=2, BW=32, MEM_L=10
    echo ""
    echo "[REPORT] PipeANN(mode=2) sel=10% T=${T} bigann1B"
    log "BigANN-1B PipeANN T=${T}..."
    for L in $L_VALUES; do
        run_search "$DTYPE" "$INDEX" "$T" 32 \
            "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
            2 10 0 $L
    done

    # GateANN: mode=8, BW=32, MEM_L=10, full_adj_nbrs=16
    echo ""
    echo "[REPORT] GateANN(mode=8) sel=10% T=${T} bigann1B"
    log "BigANN-1B GateANN T=${T}..."
    for L in $L_VALUES; do
        run_search "$DTYPE" "$INDEX" "$T" 32 \
            "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
            8 10 0 16 $L
    done
done
} > "$OUTFILE"

log "BigANN-1B Pareto done -> $OUTFILE"
log "=== fig08_billion.sh COMPLETE ==="
