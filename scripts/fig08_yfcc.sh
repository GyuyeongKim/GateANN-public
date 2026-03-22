#!/bin/bash
set -e

# Figure 8: YFCC-10M multi-label filtered search
# PipeANN (post-filter, mode=2) vs GateANN (mode=8), T=32
# Uses search_disk_index_yfcc binary (spmat metadata format)
#
# Usage: ./scripts/fig08_yfcc.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${GATEANN_DATA_DIR:-${REPO_ROOT}/data}"
INDEX_DIR="${GATEANN_INDEX_DIR:-${DATA_DIR}/pipeann_indices}"
YFCC_DIR="${DATA_DIR}/yfcc10M"
RESULTS_DIR="${REPO_ROOT}/data/filter/results"
SEARCH_BIN="${REPO_ROOT}/build/tests/search_disk_index_yfcc"

mkdir -p "$RESULTS_DIR"

# search_disk_index_yfcc arg format:
#   <type> <index_prefix> <num_threads> <beamwidth>
#   <query_bin> <base_metadata.spmat> <query_metadata.spmat> <filtered_gt.bin>
#   <K> <dist_metric> <nbr_type>
#   <mode(2=pipeann,8=gateann)> <mem_L> <cache_budget>
#   [full_adj_max_nbrs (mode=8)]
#   <L1> [L2] ...

L_VALUES="20 30 50 70 100 150 200 300 400 500"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

###############################################################################
# YFCC-10M
###############################################################################
DTYPE="uint8"
INDEX="${INDEX_DIR}/yfcc10M"
QUERY="${YFCC_DIR}/query.public.100K.u8bin"
BASE_META="${YFCC_DIR}/base.metadata.10M.spmat"
QUERY_META="${YFCC_DIR}/query.metadata.public.100K.spmat"
GT="${DATA_DIR}/filter/yfcc10M_filtered_gt.bin"

OUTFILE="${RESULTS_DIR}/fig_yfcc.txt"

{
echo "========================================"
echo " Figure 8: YFCC-10M Multi-Label (T=32)"
echo " $(date)"
echo "========================================"

# PipeANN: mode=2, BW=32, MEM_L=0
echo ""
echo "[REPORT] PipeANN(mode=2) T=32 yfcc10M"
log "YFCC PipeANN T=32..."
for L in $L_VALUES; do
    "$SEARCH_BIN" "$DTYPE" "$INDEX" 32 32 \
        "$QUERY" "$BASE_META" "$QUERY_META" "$GT" \
        10 l2 pq 2 0 0 $L 2>&1 | grep -P '^\s+\d+\s' || true
done

# GateANN: mode=8, BW=32, MEM_L=0, full_adj_nbrs=32
echo ""
echo "[REPORT] GateANN(mode=8) T=32 yfcc10M"
log "YFCC GateANN T=32..."
for L in $L_VALUES; do
    "$SEARCH_BIN" "$DTYPE" "$INDEX" 32 32 \
        "$QUERY" "$BASE_META" "$QUERY_META" "$GT" \
        10 l2 pq 8 0 0 32 $L 2>&1 | grep -P '^\s+\d+\s' || true
done
} > "$OUTFILE"

log "YFCC-10M done -> $OUTFILE"
log "=== fig08_yfcc.sh COMPLETE ==="
