#!/bin/bash
set -e

# Quick validation script (~10 min)
# Runs mode=0 (DiskANN), mode=2 (PipeANN), mode=8 (GateANN) on BigANN-100M
# with T=1, L=50,100,200 and verifies GateANN > PipeANN > DiskANN in QPS.
#
# Usage: ./scripts/quick_validate.sh
#
# Env vars:
#   GATEANN_DATA_DIR — root data directory (default: <repo>/data)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${GATEANN_DATA_DIR:-${REPO_ROOT}/data}"
INDEX_DIR="${GATEANN_INDEX_DIR:-${DATA_DIR}/pipeann_indices}"
FILTER_DIR="${DATA_DIR}/filter_exp_100M"
RESULTS_DIR="${REPO_ROOT}/data/filter/results"
SEARCH_BIN="${REPO_ROOT}/build/tests/search_disk_index_fa"

mkdir -p "$RESULTS_DIR"

L_VALUES="50 100 200"
T=1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

###############################################################################
# Pre-flight checks
###############################################################################
if [ ! -x "$SEARCH_BIN" ]; then
    echo "ERROR: Search binary not found: $SEARCH_BIN"
    echo "Build first: cd $REPO_ROOT && mkdir -p build && cd build && cmake .. && make -j"
    exit 1
fi

DTYPE="uint8"
INDEX="${INDEX_DIR}/bigann100M"
QUERY="${DATA_DIR}/bigann100M_query.u8bin"
NLABELS="${FILTER_DIR}/bigann100M_node_labels.bin"
QLABELS="${FILTER_DIR}/bigann100M_query_labels.bin"
GT="${FILTER_DIR}/bigann100M_filtered_gt.bin"

MISSING=0
for f in "${INDEX}_disk.index" "${INDEX}_pq_compressed.bin" \
         "$QUERY" "$NLABELS" "$QLABELS" "$GT"; do
    if [ ! -f "$f" ]; then
        echo "MISSING: $f"
        MISSING=1
    fi
done
if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "Required files missing. Run setup_data.sh bigann100M first,"
    echo "then build the index and generate ground truth."
    exit 1
fi

log "=== Quick Validation Start ==="
log "Dataset: BigANN-100M, T=${T}, L=${L_VALUES}"

OUTFILE="${RESULTS_DIR}/quick_validate.txt"
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

###############################################################################
# Run experiments
###############################################################################
run_search() {
    local mode_args="$@"
    "$SEARCH_BIN" "$DTYPE" "$INDEX" "$T" "$BW" \
        "$QUERY" "$NLABELS" "$QLABELS" "$GT" \
        10 l2 pq $mode_args 2>&1 | grep -P '^\s+\d+\s' || true
}

# DiskANN (mode=0, BW=8)
log "Running DiskANN (mode=0, BW=8)..."
BW=8
{
echo "[REPORT] DiskANN(mode=0) sel=10% T=${T} bigann100M"
for L in $L_VALUES; do
    run_search 0 0 0 $L
done
} > "$TMPDIR/diskann.txt"

# PipeANN (mode=2, BW=32)
log "Running PipeANN (mode=2, BW=32)..."
BW=32
{
echo "[REPORT] PipeANN(mode=2) sel=10% T=${T} bigann100M"
for L in $L_VALUES; do
    run_search 2 10 0 $L
done
} > "$TMPDIR/pipeann.txt"

# GateANN (mode=8, BW=32, nbrs=32)
log "Running GateANN (mode=8, BW=32, nbrs=32)..."
BW=32
{
echo "[REPORT] GateANN(mode=8) sel=10% T=${T} bigann100M"
for L in $L_VALUES; do
    run_search 8 10 0 32 $L
done
} > "$TMPDIR/gateann.txt"

###############################################################################
# Print comparison table
###############################################################################
{
echo "========================================"
echo " Quick Validation Results (BigANN-100M, T=${T}, sel=10%)"
echo " $(date)"
echo "========================================"
echo ""

cat "$TMPDIR/diskann.txt"
echo ""
cat "$TMPDIR/pipeann.txt"
echo ""
cat "$TMPDIR/gateann.txt"
} | tee "$OUTFILE"

###############################################################################
# Verify ordering: GateANN QPS > PipeANN QPS > DiskANN QPS (at each L)
###############################################################################
echo ""
echo "========================================"
echo " Validation Check"
echo "========================================"

PASS=0
FAIL=0
WARN=0

python3 - "$TMPDIR/diskann.txt" "$TMPDIR/pipeann.txt" "$TMPDIR/gateann.txt" <<'PYEOF'
import sys, re

def parse_qps(filepath):
    results = {}
    with open(filepath) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    L = int(parts[0])
                    qps = float(parts[1])
                    recall = float(parts[2])
                    results[L] = (qps, recall)
                except ValueError:
                    pass
    return results

diskann = parse_qps(sys.argv[1])
pipeann = parse_qps(sys.argv[2])
gateann = parse_qps(sys.argv[3])

print(f"{'L':>6}  {'DiskANN QPS':>12} {'PipeANN QPS':>12} {'GateANN QPS':>12}  {'Status':>8}")
print("-" * 66)

pass_count = 0
fail_count = 0

for L in sorted(set(diskann.keys()) & set(pipeann.keys()) & set(gateann.keys())):
    d_qps = diskann[L][0]
    p_qps = pipeann[L][0]
    g_qps = gateann[L][0]

    ok = g_qps > p_qps > d_qps
    status = "PASS" if ok else "FAIL"
    if ok:
        pass_count += 1
    else:
        fail_count += 1

    print(f"{L:>6}  {d_qps:>12.1f} {p_qps:>12.1f} {g_qps:>12.1f}  {status:>8}")

print()
if fail_count == 0:
    print(f"ALL PASSED ({pass_count}/{pass_count}): GateANN QPS > PipeANN QPS > DiskANN QPS")
else:
    print(f"FAILED: {fail_count} L values did not satisfy GateANN > PipeANN > DiskANN")
    print("This may be expected at very small L values where overhead dominates.")

sys.exit(0 if fail_count == 0 else 1)
PYEOF

RESULT=$?

echo ""
if [ $RESULT -eq 0 ]; then
    log "=== VALIDATION PASSED ==="
else
    log "=== VALIDATION FAILED (see above) ==="
fi

log "Results saved to: $OUTFILE"
log "=== quick_validate.sh COMPLETE ==="
exit $RESULT
