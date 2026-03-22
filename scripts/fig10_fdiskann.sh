#!/bin/bash
set -e

# Figure 10: Filtered-DiskANN comparison — BigANN-100M + DEEP-100M
#
# Uses the OFFICIAL DiskANN repository (https://github.com/microsoft/DiskANN)
# for building and searching FilteredVamana indices.
# GateANN/PipeANN/DiskANN use our binary on the unmodified Vamana graph.
#
# Prerequisites:
#   1. Clone and build official DiskANN:
#        git clone https://github.com/microsoft/DiskANN.git
#        cd DiskANN && mkdir build && cd build && cmake .. && make -j
#   2. Set DISKANN_BUILD_DIR to the DiskANN build directory:
#        export DISKANN_BUILD_DIR=/path/to/DiskANN/build
#
# Usage: ./scripts/fig10_fdiskann.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${GATEANN_DATA_DIR:-${REPO_ROOT}/data}"
INDEX_DIR="${GATEANN_INDEX_DIR:-${DATA_DIR}/pipeann_indices}"
FILTER_DIR="${DATA_DIR}/filter_exp_100M"
RESULTS_DIR="${REPO_ROOT}/data/filter/results"
SEARCH_BIN="${REPO_ROOT}/build/tests/search_disk_index_fa"

mkdir -p "$RESULTS_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

###############################################################################
# Validate official DiskANN
###############################################################################
if [ -z "$DISKANN_BUILD_DIR" ]; then
    echo "ERROR: Set DISKANN_BUILD_DIR to the official DiskANN build directory."
    echo ""
    echo "  git clone https://github.com/microsoft/DiskANN.git"
    echo "  cd DiskANN && mkdir build && cd build && cmake .. && make -j"
    echo "  export DISKANN_BUILD_DIR=\$(pwd)"
    exit 1
fi

DISKANN_BUILD_BIN="${DISKANN_BUILD_DIR}/apps/build_disk_index"
DISKANN_SEARCH_BIN="${DISKANN_BUILD_DIR}/apps/search_disk_index"

for bin in "$DISKANN_BUILD_BIN" "$DISKANN_SEARCH_BIN"; do
    if [ ! -x "$bin" ]; then
        echo "ERROR: Not found: $bin"
        echo "Build official DiskANN first."
        exit 1
    fi
done

L_VALUES="20 30 40 50 70 100 150 200 250 300 400 500 700 1000 1500 2000"

###############################################################################
# GateANN helper (our binary)
###############################################################################
run_our_search() {
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
BIGANN_DTYPE="uint8"
BIGANN_INDEX="${INDEX_DIR}/bigann100M"
BIGANN_QUERY="${DATA_DIR}/bigann100M_query.u8bin"
BIGANN_NLABELS="${FILTER_DIR}/bigann100M_node_labels.bin"
BIGANN_QLABELS="${FILTER_DIR}/bigann100M_query_labels.bin"
BIGANN_GT="${FILTER_DIR}/bigann100M_filtered_gt.bin"
BIGANN_NODE_LABELS_TXT="${FILTER_DIR}/bigann100M_node_labels.txt"
BIGANN_QUERY_FILTERS_TXT="${FILTER_DIR}/bigann100M_query_filters.txt"
BIGANN_BASE="${DATA_DIR}/bigann100M_base.u8bin"

# --- Build FilteredVamana index if not exists ---
BIGANN_FILTERED_PREFIX="${INDEX_DIR}/bigann100M_official_filtered"
if [ ! -f "${BIGANN_FILTERED_PREFIX}_disk.index" ]; then
    log "Building BigANN-100M FilteredVamana index (official DiskANN)..."
    log "This may take several hours."
    if [ ! -f "$BIGANN_BASE" ]; then
        echo "ERROR: Base vectors not found: $BIGANN_BASE"
        exit 1
    fi
    if [ ! -f "$BIGANN_NODE_LABELS_TXT" ]; then
        echo "ERROR: Text label file not found: $BIGANN_NODE_LABELS_TXT"
        echo "Run ./scripts/setup_data.sh bigann100M first."
        exit 1
    fi
    $DISKANN_BUILD_BIN --data_type uint8 --dist_fn l2 \
        --data_path "$BIGANN_BASE" \
        --index_path_prefix "$BIGANN_FILTERED_PREFIX" \
        -R 96 --FilteredLbuild 100 -B 4 -M 80 -T 64 --PQ_disk_bytes 0 \
        --label_file "$BIGANN_NODE_LABELS_TXT" \
        2>&1 | tee "${RESULTS_DIR}/bigann100M_official_filtered_build.log"
    log "BigANN-100M FilteredVamana build complete."
else
    log "BigANN-100M FilteredVamana index found: ${BIGANN_FILTERED_PREFIX}_disk.index"
fi

BIGANN_OUT="${RESULTS_DIR}/fig_fdiskann_bigann100M.txt"

{
echo "========================================"
echo " Figure 10: Filtered-DiskANN vs GateANN (BigANN-100M, sel=10%)"
echo " $(date)"
echo "========================================"

for T in 1 32; do
    # Filtered-DiskANN: official DiskANN search binary
    echo ""
    echo "[REPORT] FilteredDiskANN sel=10% T=${T} bigann100M"
    log "Filtered-DiskANN (official) T=${T}..."
    $DISKANN_SEARCH_BIN --data_type uint8 --dist_fn l2 \
        --index_path_prefix "$BIGANN_FILTERED_PREFIX" \
        --query_file "$BIGANN_QUERY" \
        --gt_file "$BIGANN_GT" \
        --query_filters_file "$BIGANN_QUERY_FILTERS_TXT" \
        -K 10 -L $L_VALUES \
        -W 8 -T $T \
        --result_path /tmp/bigann_fdiskann_T${T} 2>&1

    # DiskANN: mode=0, BW=8 (post-filter baseline)
    echo ""
    echo "[REPORT] DiskANN(mode=0) sel=10% T=${T} bigann100M"
    log "DiskANN T=${T}..."
    for L in $L_VALUES; do
        run_our_search "$BIGANN_DTYPE" "$BIGANN_INDEX" "$T" 8 \
            "$BIGANN_QUERY" "$BIGANN_NLABELS" "$BIGANN_QLABELS" "$BIGANN_GT" \
            0 0 0 $L
    done

    # PipeANN: mode=2, BW=32
    echo ""
    echo "[REPORT] PipeANN(mode=2) sel=10% T=${T} bigann100M"
    log "PipeANN T=${T}..."
    for L in $L_VALUES; do
        run_our_search "$BIGANN_DTYPE" "$BIGANN_INDEX" "$T" 32 \
            "$BIGANN_QUERY" "$BIGANN_NLABELS" "$BIGANN_QLABELS" "$BIGANN_GT" \
            2 10 0 $L
    done

    # GateANN: mode=8, BW=32, nbrs=32
    echo ""
    echo "[REPORT] GateANN(mode=8) sel=10% T=${T} bigann100M"
    log "GateANN T=${T}..."
    for L in $L_VALUES; do
        run_our_search "$BIGANN_DTYPE" "$BIGANN_INDEX" "$T" 32 \
            "$BIGANN_QUERY" "$BIGANN_NLABELS" "$BIGANN_QLABELS" "$BIGANN_GT" \
            8 10 0 32 $L
    done
done
} > "$BIGANN_OUT"

log "BigANN-100M Filtered-DiskANN comparison done -> $BIGANN_OUT"

###############################################################################
# DEEP-100M
###############################################################################
DEEP_DTYPE="float"
DEEP_INDEX="${INDEX_DIR}/deep100M"
DEEP_QUERY="${DATA_DIR}/deep100M_query.fbin"
DEEP_NLABELS="${FILTER_DIR}/deep100M_node_labels.bin"
DEEP_QLABELS="${FILTER_DIR}/deep100M_query_labels.bin"
DEEP_GT="${FILTER_DIR}/deep100M_filtered_gt.bin"
DEEP_NODE_LABELS_TXT="${FILTER_DIR}/deep100M_node_labels.txt"
DEEP_QUERY_FILTERS_TXT="${FILTER_DIR}/deep100M_query_filters.txt"
DEEP_BASE="${DATA_DIR}/deep100M_base.fbin"

# --- Build FilteredVamana index if not exists ---
DEEP_FILTERED_PREFIX="${INDEX_DIR}/deep100M_official_filtered"
if [ ! -f "${DEEP_FILTERED_PREFIX}_disk.index" ]; then
    log "Building DEEP-100M FilteredVamana index (official DiskANN)..."
    if [ ! -f "$DEEP_BASE" ]; then
        echo "WARNING: Base vectors not found: $DEEP_BASE — skipping DEEP-100M."
        exit 0
    fi
    $DISKANN_BUILD_BIN --data_type float --dist_fn l2 \
        --data_path "$DEEP_BASE" \
        --index_path_prefix "$DEEP_FILTERED_PREFIX" \
        -R 96 --FilteredLbuild 128 -B 4 -M 80 -T 64 --PQ_disk_bytes 0 \
        --label_file "$DEEP_NODE_LABELS_TXT" \
        2>&1 | tee "${RESULTS_DIR}/deep100M_official_filtered_build.log"
    log "DEEP-100M FilteredVamana build complete."
else
    log "DEEP-100M FilteredVamana index found: ${DEEP_FILTERED_PREFIX}_disk.index"
fi

DEEP_OUT="${RESULTS_DIR}/fig_fdiskann_deep100M.txt"

{
echo "========================================"
echo " Figure 10: Filtered-DiskANN vs GateANN (DEEP-100M, sel=10%)"
echo " $(date)"
echo "========================================"

for T in 1 32; do
    # Filtered-DiskANN: official DiskANN search
    echo ""
    echo "[REPORT] FilteredDiskANN sel=10% T=${T} deep100M"
    log "Filtered-DiskANN (official, DEEP) T=${T}..."
    $DISKANN_SEARCH_BIN --data_type float --dist_fn l2 \
        --index_path_prefix "$DEEP_FILTERED_PREFIX" \
        --query_file "$DEEP_QUERY" \
        --gt_file "$DEEP_GT" \
        --query_filters_file "$DEEP_QUERY_FILTERS_TXT" \
        -K 10 -L $L_VALUES \
        -W 8 -T $T \
        --result_path /tmp/deep_fdiskann_T${T} 2>&1

    # PipeANN
    echo ""
    echo "[REPORT] PipeANN(mode=2) sel=10% T=${T} deep100M"
    log "PipeANN (DEEP) T=${T}..."
    for L in $L_VALUES; do
        run_our_search "$DEEP_DTYPE" "$DEEP_INDEX" "$T" 32 \
            "$DEEP_QUERY" "$DEEP_NLABELS" "$DEEP_QLABELS" "$DEEP_GT" \
            2 10 0 $L
    done

    # GateANN
    echo ""
    echo "[REPORT] GateANN(mode=8) sel=10% T=${T} deep100M"
    log "GateANN (DEEP) T=${T}..."
    for L in $L_VALUES; do
        run_our_search "$DEEP_DTYPE" "$DEEP_INDEX" "$T" 32 \
            "$DEEP_QUERY" "$DEEP_NLABELS" "$DEEP_QLABELS" "$DEEP_GT" \
            8 10 0 32 $L
    done
done
} > "$DEEP_OUT"

log "DEEP-100M Filtered-DiskANN comparison done -> $DEEP_OUT"
log "=== fig10_fdiskann.sh COMPLETE ==="
