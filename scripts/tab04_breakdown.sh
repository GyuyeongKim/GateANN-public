#!/bin/bash
set -e

# Table 4: Time breakdown — IO vs Tunneling vs Processing for GateANN
#
# Extracts per-phase timing from search output (bd_io_us, bd_tunnel_us, bd_process_us).
#
# Usage: ./scripts/tab04_breakdown.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${GATEANN_DATA_DIR:-${REPO_ROOT}/data}"
INDEX_DIR="${GATEANN_INDEX_DIR:-${DATA_DIR}/pipeann_indices}"
FILTER_DIR="${DATA_DIR}/filter_exp_100M"
RESULTS_DIR="${REPO_ROOT}/data/filter/results"
SEARCH_BIN="${REPO_ROOT}/build/tests/search_disk_index_fa"

mkdir -p "$RESULTS_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

DTYPE="uint8"
INDEX="${INDEX_DIR}/bigann100M"
QUERY="${DATA_DIR}/bigann100M_query.u8bin"
NLABELS="${FILTER_DIR}/bigann100M_node_labels.bin"
QLABELS="${FILTER_DIR}/bigann100M_query_labels.bin"
GT="${FILTER_DIR}/bigann100M_filtered_gt.bin"

L_VALUES="50 100 200 300 500"
T=1

OUTFILE="${RESULTS_DIR}/tab_breakdown.txt"

{
echo "========================================"
echo " Table 4: Time Breakdown (BigANN-100M, sel=10%, T=${T})"
echo " $(date)"
echo "========================================"
echo ""
echo "Columns: L  QPS  Recall  MeanIOs  FilterSkips  IO_us  Tunnel_us  Process_us  RSS_GB"
echo ""

# PipeANN baseline (for comparison)
echo "[REPORT] PipeANN(mode=2) sel=10% T=${T} bigann100M"
log "PipeANN..."
for L in $L_VALUES; do
    "$SEARCH_BIN" "$DTYPE" "$INDEX" "$T" 32 \
        "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
        10 l2 pq 2 10 0 $L 2>&1 | grep -P '^\s+\d+\s' || true
done

echo ""

# GateANN (mode=8) — includes IO, tunnel, process breakdown
echo "[REPORT] GateANN(mode=8) sel=10% T=${T} bigann100M"
log "GateANN..."
for L in $L_VALUES; do
    "$SEARCH_BIN" "$DTYPE" "$INDEX" "$T" 32 \
        "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
        10 l2 pq 8 10 0 32 $L 2>&1 | grep -P '^\s+\d+\s' || true
done
} > "$OUTFILE"

cat "$OUTFILE"

log "Breakdown done -> $OUTFILE"
log "=== tab04_breakdown.sh COMPLETE ==="
