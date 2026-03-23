#!/bin/bash
set -e

# Table 3: SSD speed impact — compare GateANN on two SSDs with different speeds
#
# This experiment measures how SSD speed affects the throughput gap between
# GateANN and PipeANN. Since GateANN eliminates most SSD I/Os, it benefits
# less from faster SSDs, and conversely, its advantage grows on slower SSDs.
#
# Prerequisites:
#   - The same BigANN-100M index must exist on BOTH SSDs.
#   - Set environment variables to point to each SSD's index prefix.
#
# Usage:
#   export GATEANN_SSD1_INDEX=/fast_ssd/pipeann_indices/bigann100M
#   export GATEANN_SSD2_INDEX=/slow_ssd/pipeann_indices/bigann100M
#   export GATEANN_SSD1_LABEL="Samsung 990 Pro (7 GB/s)"  # optional
#   export GATEANN_SSD2_LABEL="Intel 660p (1.5 GB/s)"     # optional
#   ./scripts/tab04_ssd_impact.sh
#
# To copy the index to a second SSD:
#   cp /ssd1/pipeann_indices/bigann100M* /ssd2/pipeann_indices/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${GATEANN_DATA_DIR:-${REPO_ROOT}/data}"
FILTER_DIR="${DATA_DIR}/filter_exp_100M"
RESULTS_DIR="${REPO_ROOT}/data/filter/results"
SEARCH_BIN="${REPO_ROOT}/build/tests/search_disk_index_fa"

mkdir -p "$RESULTS_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

###############################################################################
# Validate environment
###############################################################################
if [ -z "$GATEANN_SSD1_INDEX" ] || [ -z "$GATEANN_SSD2_INDEX" ]; then
    echo "ERROR: Set GATEANN_SSD1_INDEX and GATEANN_SSD2_INDEX to index prefixes on two different SSDs."
    echo ""
    echo "Example:"
    echo "  export GATEANN_SSD1_INDEX=/fast_ssd/pipeann_indices/bigann100M"
    echo "  export GATEANN_SSD2_INDEX=/slow_ssd/pipeann_indices/bigann100M"
    echo "  ./scripts/tab04_ssd_impact.sh"
    exit 1
fi

SSD1_LABEL="${GATEANN_SSD1_LABEL:-SSD1}"
SSD2_LABEL="${GATEANN_SSD2_LABEL:-SSD2}"

# Check that index files exist on both SSDs
for ssd_index in "$GATEANN_SSD1_INDEX" "$GATEANN_SSD2_INDEX"; do
    for suffix in "_disk.index" "_pq_compressed.bin"; do
        if [ ! -f "${ssd_index}${suffix}" ]; then
            echo "ERROR: Missing ${ssd_index}${suffix}"
            echo "Copy the index to this SSD first."
            exit 1
        fi
    done
done

# Check that filter/query data exist
DTYPE="uint8"
QUERY="${DATA_DIR}/bigann100M_query.u8bin"
NLABELS="${FILTER_DIR}/bigann100M_node_labels.bin"
QLABELS="${FILTER_DIR}/bigann100M_query_labels.bin"
GT="${FILTER_DIR}/bigann100M_filtered_gt.bin"

for f in "$QUERY" "$NLABELS" "$QLABELS" "$GT"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"
        exit 1
    fi
done

# Verify the two paths are on different devices
SSD1_DEV=$(df "$GATEANN_SSD1_INDEX" 2>/dev/null | tail -1 | awk '{print $1}')
SSD2_DEV=$(df "$GATEANN_SSD2_INDEX" 2>/dev/null | tail -1 | awk '{print $1}')
if [ "$SSD1_DEV" = "$SSD2_DEV" ]; then
    echo "WARNING: Both index paths appear to be on the same device ($SSD1_DEV)."
    echo "         Results will not reflect different SSD speeds."
    echo ""
fi

###############################################################################
# Helpers
###############################################################################
L_VALUES="50 100 200 300 500"
T=1

run_search() {
    local index=$1
    shift
    local mode_args="$@"

    "$SEARCH_BIN" "$DTYPE" "$index" "$T" "$BW" \
        "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
        10 l2 pq $mode_args 2>&1 | grep -P '^\s+\d+\s' || true
}

###############################################################################
# Run experiments on each SSD
###############################################################################
OUTFILE="${RESULTS_DIR}/tab_ssd_impact.txt"

{
echo "========================================"
echo " Table 3: SSD Speed Impact (BigANN-100M, sel=10%, T=${T})"
echo " SSD1: ${SSD1_LABEL} (${SSD1_DEV})"
echo " SSD2: ${SSD2_LABEL} (${SSD2_DEV})"
echo " $(date)"
echo "========================================"

for SSD_NAME in "SSD1" "SSD2"; do
    if [ "$SSD_NAME" = "SSD1" ]; then
        INDEX="$GATEANN_SSD1_INDEX"
        LABEL="$SSD1_LABEL"
    else
        INDEX="$GATEANN_SSD2_INDEX"
        LABEL="$SSD2_LABEL"
    fi

    # PipeANN (mode=2, BW=32)
    echo ""
    echo "[REPORT] PipeANN(mode=2) sel=10% T=${T} ${SSD_NAME}(${LABEL})"
    log "${SSD_NAME} PipeANN..."
    BW=32
    for L in $L_VALUES; do
        run_search "$INDEX" 2 10 0 $L
    done

    # GateANN (mode=8, BW=32, nbrs=32)
    echo ""
    echo "[REPORT] GateANN(mode=8) sel=10% T=${T} ${SSD_NAME}(${LABEL})"
    log "${SSD_NAME} GateANN..."
    BW=32
    for L in $L_VALUES; do
        run_search "$INDEX" 8 10 0 32 $L
    done
done
} > "$OUTFILE"

log "SSD impact done -> $OUTFILE"
log "=== tab04_ssd_impact.sh COMPLETE ==="
