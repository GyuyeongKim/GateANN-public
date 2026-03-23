#!/bin/bash
set -e

# Dataset download and label generation helper for GateANN artifact.
#
# Usage:
#   ./scripts/setup_data.sh bigann100M   # BigANN-100M (uint8, 128d)
#   ./scripts/setup_data.sh deep100M     # DEEP-100M   (float, 96d)
#   ./scripts/setup_data.sh yfcc10M      # YFCC-10M    (uint8, 192d)
#   ./scripts/setup_data.sh bigann1B     # BigANN-1B   (uint8, 128d)
#
# Env vars (override defaults):
#   GATEANN_DATA_DIR — root data directory (default: <repo>/data)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${GATEANN_DATA_DIR:-${REPO_ROOT}/data}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

usage() {
    echo "Usage: $0 {bigann100M|deep100M|yfcc10M|bigann1B}"
    exit 1
}

###############################################################################
# Label generation helper (uniform random, 10 classes, ~10% per class)
###############################################################################
generate_uniform_labels() {
    local npts=$1 nqueries=$2 nclasses=$3 prefix=$4 outdir=$5
    mkdir -p "$outdir"

    python3 - "$npts" "$nqueries" "$nclasses" "$prefix" "$outdir" <<'PYEOF'
import struct, os, sys
import numpy as np

npts      = int(sys.argv[1])
nqueries  = int(sys.argv[2])
nclasses  = int(sys.argv[3])
prefix    = sys.argv[4]
outdir    = sys.argv[5]

np.random.seed(42)

# Node labels: [uint32 n][uint8 label[n]]
node_labels = np.random.randint(0, nclasses, size=npts, dtype=np.uint8)
node_path = os.path.join(outdir, f"{prefix}_node_labels.bin")
with open(node_path, 'wb') as f:
    f.write(struct.pack('I', npts))
    f.write(node_labels.tobytes())
print(f"  Wrote {node_path}  ({npts} nodes, {nclasses} classes)")

# Query labels: [uint32 n][uint8 label[n]]
query_labels = np.random.randint(0, nclasses, size=nqueries, dtype=np.uint8)
query_path = os.path.join(outdir, f"{prefix}_query_labels.bin")
with open(query_path, 'wb') as f:
    f.write(struct.pack('I', nqueries))
    f.write(query_labels.tobytes())
print(f"  Wrote {query_path}  ({nqueries} queries)")

# Text format for official DiskANN (one label per line)
node_txt_path = os.path.join(outdir, f"{prefix}_node_labels.txt")
with open(node_txt_path, 'w') as f:
    for lbl in node_labels:
        f.write(f"{lbl}\n")
print(f"  Wrote {node_txt_path}  (text format for official DiskANN)")

query_txt_path = os.path.join(outdir, f"{prefix}_query_filters.txt")
with open(query_txt_path, 'w') as f:
    for lbl in query_labels:
        f.write(f"{lbl}\n")
print(f"  Wrote {query_txt_path}  (text format for official DiskANN)")

print(f"  NOTE: Ground truth must be computed separately with the built index.")
PYEOF
}

###############################################################################
# BigANN-100M
###############################################################################
setup_bigann100M() {
    log "=== Setting up BigANN-100M ==="

    local BASEDIR="${DATA_DIR}"
    local FILTER_DIR="${DATA_DIR}/filter_exp_100M"
    mkdir -p "$BASEDIR" "$FILTER_DIR"

    # Download base vectors
    local BASE_FILE="$BASEDIR/bigann100M_base.u8bin"
    if [ ! -f "$BASE_FILE" ]; then
        log "Downloading BigANN-100M base vectors..."
        echo "  Source: https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin"
        echo "  This file is ~128GB for the full 1B; we need the first 100M vectors."
        echo ""
        echo "  Option A: Download full file and truncate:"
        echo "    wget -O ${BASEDIR}/bigann1B_base.u8bin https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin"
        echo "    # Extract first 100M vectors (8 bytes header + 100M*128 bytes):"
        echo "    head -c 12800000008 ${BASEDIR}/bigann1B_base.u8bin > $BASE_FILE"
        echo ""
        echo "  Option B: If you have the file locally, symlink it:"
        echo "    ln -s /path/to/bigann100M_base.u8bin $BASE_FILE"
        echo ""
    else
        log "Base vectors found: $BASE_FILE"
    fi

    # Download query vectors
    local QUERY_FILE="$BASEDIR/bigann100M_query.u8bin"
    if [ ! -f "$QUERY_FILE" ]; then
        log "Downloading BigANN query vectors..."
        echo "  wget -O $QUERY_FILE https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin"
        echo ""
    else
        log "Query vectors found: $QUERY_FILE"
    fi

    # Generate uniform labels (10 classes, ~10% selectivity)
    if [ ! -f "${FILTER_DIR}/bigann100M_node_labels.bin" ]; then
        log "Generating uniform labels (10 classes)..."
        generate_uniform_labels 100000000 10000 10 "bigann100M" "$FILTER_DIR"
    else
        log "Labels found: ${FILTER_DIR}/bigann100M_node_labels.bin"
    fi

    echo ""
    log "=== BigANN-100M Setup Summary ==="
    echo "  Base:   $BASE_FILE"
    echo "  Query:  $QUERY_FILE"
    echo "  Labels: ${FILTER_DIR}/bigann100M_node_labels.bin"
    echo ""
    echo "  Next steps:"
    echo "  1. Build index:"
    echo "     ./build/tests/build_disk_index uint8 $BASE_FILE ${DATA_DIR}/pipeann_indices/bigann100M 128 200 32 80 64 l2 pq"
    echo "  2. Generate ground truth (filtered):"
    echo "     (see README.md for instructions)"
    echo "  3. Run experiments: ./scripts/fig05_pareto_main.sh"
}

###############################################################################
# DEEP-100M
###############################################################################
setup_deep100M() {
    log "=== Setting up DEEP-100M ==="

    local BASEDIR="${DATA_DIR}"
    local FILTER_DIR="${DATA_DIR}/filter_exp_100M"
    mkdir -p "$BASEDIR" "$FILTER_DIR"

    local BASE_FILE="$BASEDIR/deep100M_base.fbin"
    if [ ! -f "$BASE_FILE" ]; then
        log "DEEP-100M base vectors needed."
        echo "  Source: https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/deep1B/base.1B.fbin"
        echo "  Download first 100M vectors or the full 1B and truncate."
        echo "  Expected format: .fbin (float32, 96d)"
        echo ""
    else
        log "Base vectors found: $BASE_FILE"
    fi

    local QUERY_FILE="$BASEDIR/deep100M_query.fbin"
    if [ ! -f "$QUERY_FILE" ]; then
        log "DEEP query vectors needed."
        echo "  wget -O $QUERY_FILE https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/deep1B/query.public.10K.fbin"
        echo ""
    else
        log "Query vectors found: $QUERY_FILE"
    fi

    if [ ! -f "${FILTER_DIR}/deep100M_node_labels.bin" ]; then
        log "Generating uniform labels (10 classes)..."
        generate_uniform_labels 100000000 10000 10 "deep100M" "$FILTER_DIR"
    else
        log "Labels found: ${FILTER_DIR}/deep100M_node_labels.bin"
    fi

    echo ""
    log "=== DEEP-100M Setup Summary ==="
    echo "  Base:   $BASE_FILE"
    echo "  Query:  $QUERY_FILE"
    echo "  Labels: ${FILTER_DIR}/deep100M_node_labels.bin"
    echo ""
    echo "  Next steps:"
    echo "  1. Build index:"
    echo "     ./build/tests/build_disk_index float $BASE_FILE ${DATA_DIR}/pipeann_indices/deep100M 128 200 32 80 64 l2 pq"
    echo "  2. Generate ground truth, then run experiments."
}

###############################################################################
# YFCC-10M
###############################################################################
setup_yfcc10M() {
    log "=== Setting up YFCC-10M ==="

    local YFCC_DIR="${DATA_DIR}/yfcc10M"
    mkdir -p "$YFCC_DIR"

    local BASE_FILE="$YFCC_DIR/base.10M.u8bin"
    if [ ! -f "$BASE_FILE" ]; then
        log "YFCC-10M base vectors needed."
        echo "  The YFCC-10M dataset requires:"
        echo "    - base.10M.u8bin            (base vectors, uint8, 192d)"
        echo "    - query.public.100K.u8bin    (query vectors)"
        echo "    - base.metadata.10M.spmat    (multi-label metadata, sparse matrix format)"
        echo "    - query.metadata.public.100K.spmat (query metadata)"
        echo ""
        echo "  These can be obtained from the YFCC100M dataset."
        echo "  Place files in: $YFCC_DIR/"
        echo ""
    else
        log "YFCC base vectors found: $BASE_FILE"
    fi

    echo ""
    log "=== YFCC-10M Setup Summary ==="
    echo "  Directory: $YFCC_DIR"
    echo "  Required files:"
    echo "    - base.10M.u8bin"
    echo "    - query.public.100K.u8bin"
    echo "    - base.metadata.10M.spmat"
    echo "    - query.metadata.public.100K.spmat"
    echo "  Ground truth: ${DATA_DIR}/filter/yfcc10M_filtered_gt.bin"
    echo ""
    echo "  Next steps:"
    echo "  1. Build index:"
    echo "     ./build/tests/build_disk_index uint8 $BASE_FILE ${DATA_DIR}/pipeann_indices/yfcc10M 128 200 32 80 64 l2 pq"
    echo "  2. Generate filtered GT, then run: ./scripts/fig09_yfcc.sh"
}

###############################################################################
# BigANN-1B
###############################################################################
setup_bigann1B() {
    log "=== Setting up BigANN-1B ==="

    local BASEDIR="${DATA_DIR}"
    local FILTER_DIR="${DATA_DIR}/filter_exp_1B"
    mkdir -p "$BASEDIR" "$FILTER_DIR"

    local BASE_FILE="$BASEDIR/bigann1B_base.u8bin"
    if [ ! -f "$BASE_FILE" ]; then
        log "Downloading BigANN-1B base vectors..."
        echo "  wget -O $BASE_FILE https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin"
        echo "  (approximately 128 GB)"
        echo ""
    else
        log "Base vectors found: $BASE_FILE"
    fi

    local QUERY_FILE="$BASEDIR/bigann1B_query.u8bin"
    if [ ! -f "$QUERY_FILE" ]; then
        log "BigANN-1B query vectors needed."
        echo "  wget -O $QUERY_FILE https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin"
        echo ""
    else
        log "Query vectors found: $QUERY_FILE"
    fi

    if [ ! -f "${FILTER_DIR}/bigann1B_node_labels.bin" ]; then
        log "Generating uniform labels (10 classes, 1B nodes)..."
        generate_uniform_labels 1000000000 10000 10 "bigann1B" "$FILTER_DIR"
    else
        log "Labels found: ${FILTER_DIR}/bigann1B_node_labels.bin"
    fi

    echo ""
    log "=== BigANN-1B Setup Summary ==="
    echo "  Base:   $BASE_FILE"
    echo "  Query:  $QUERY_FILE"
    echo "  Labels: ${FILTER_DIR}/bigann1B_node_labels.bin"
    echo ""
    echo "  Next steps:"
    echo "  1. Build index (expect 2-3 days):"
    echo "     ./build/tests/build_disk_index uint8 $BASE_FILE ${DATA_DIR}/pipeann_indices/bigann1B 128 200 32 80 60 l2 pq"
    echo "  2. Generate filtered GT, then run: ./scripts/fig08_billion.sh"
}

###############################################################################
# Main
###############################################################################
if [ $# -lt 1 ]; then
    usage
fi

case "$1" in
    bigann100M) setup_bigann100M ;;
    deep100M)   setup_deep100M ;;
    yfcc10M)    setup_yfcc10M ;;
    bigann1B)   setup_bigann1B ;;
    *)          echo "Unknown dataset: $1"; usage ;;
esac
