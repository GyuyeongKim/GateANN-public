#!/usr/bin/env python3
"""
Q3: Generate range-predicate labels on BigANN-100M.
Uses L2 norm of each vector as a "timestamp" proxy, then discretizes
into 10 equal-frequency bins → 10% selectivity per bin.
The equality FilterStore treats bin ID as label, simulating range predicates.
"""
import struct, os, time
import numpy as np

DATA_DIR = "/Users/gykim/workspace/faiss-dev/benchmarks/data"
BASE_FILE = f"{DATA_DIR}/bigann100M_base.u8bin"
QUERY_FILE = f"{DATA_DIR}/bigann100M_query.u8bin"
GT_FILE = f"{DATA_DIR}/bigann100M_gt.bin"
OUTPUT_DIR = f"{DATA_DIR}/filter_exp_100M/range"

NUM_BINS = 10  # 10 bins → ~10% selectivity
K_OUT = 10
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(SEED)

###############################################################################
# Step 1: Compute L2 norms of all vectors (chunked)
###############################################################################
print("=== Step 1: Computing L2 norms ===")

with open(BASE_FILE, 'rb') as f:
    npts, dim = struct.unpack('II', f.read(8))
print(f"Base: {npts} pts, {dim} dim")

base_vectors = np.memmap(BASE_FILE, dtype=np.uint8, mode='r',
                          offset=8, shape=(npts, dim))

CHUNK = 10_000_000
norms = np.empty(npts, dtype=np.float32)

t0 = time.time()
for start in range(0, npts, CHUNK):
    end = min(start + CHUNK, npts)
    chunk = base_vectors[start:end].astype(np.float32)
    norms[start:end] = np.linalg.norm(chunk, axis=1)
    print(f"  {end/1e6:.0f}M / {npts/1e6:.0f}M - {time.time()-t0:.0f}s")

print(f"Norm range: [{norms.min():.1f}, {norms.max():.1f}], mean={norms.mean():.1f}")

###############################################################################
# Step 2: Equal-frequency binning → node labels
###############################################################################
print("\n=== Step 2: Binning norms into labels ===")

# Equal-frequency: each bin has exactly npts/NUM_BINS nodes
percentiles = np.linspace(0, 100, NUM_BINS + 1)
bin_edges = np.percentile(norms, percentiles)
node_labels = np.digitize(norms, bin_edges[1:-1]).astype(np.uint8)  # 0..NUM_BINS-1

for b in range(NUM_BINS):
    count = np.sum(node_labels == b)
    lo = norms[node_labels == b].min()
    hi = norms[node_labels == b].max()
    print(f"  Bin {b}: {count:,} nodes ({count/npts*100:.1f}%), norm range [{lo:.1f}, {hi:.1f}]")

np.save(os.path.join(OUTPUT_DIR, "norms.npy"), norms)
np.save(os.path.join(OUTPUT_DIR, "bin_edges.npy"), bin_edges)

###############################################################################
# Step 3: Query labels (nearest bin by query norm)
###############################################################################
print("\n=== Step 3: Query label assignment ===")

with open(QUERY_FILE, 'rb') as f:
    nq, qdim = struct.unpack('II', f.read(8))
    query_vectors = np.fromfile(f, dtype=np.uint8, count=nq * qdim).reshape(nq, qdim)

query_norms = np.linalg.norm(query_vectors.astype(np.float32), axis=1)
query_labels = np.digitize(query_norms, bin_edges[1:-1]).astype(np.uint8)

print(f"Query label distribution:")
for b in range(NUM_BINS):
    count = np.sum(query_labels == b)
    print(f"  Bin {b}: {count} queries")

###############################################################################
# Step 4: Save labels
###############################################################################
print("\n=== Step 4: Saving labels ===")

nl_path = os.path.join(OUTPUT_DIR, "node_labels_range.bin")
with open(nl_path, 'wb') as f:
    f.write(struct.pack('I', npts))
    f.write(node_labels.tobytes())
print(f"  Saved: {nl_path}")

ql_path = os.path.join(OUTPUT_DIR, "query_labels_range.bin")
with open(ql_path, 'wb') as f:
    f.write(struct.pack('I', nq))
    f.write(query_labels.tobytes())
print(f"  Saved: {ql_path}")

###############################################################################
# Step 5: Generate filtered GT
###############################################################################
print("\n=== Step 5: Filtered GT ===")

with open(GT_FILE, 'rb') as f:
    gt_nq, gt_K = struct.unpack('II', f.read(8))
    gt_ids = np.fromfile(f, dtype=np.int32, count=gt_nq * gt_K).reshape(gt_nq, gt_K)

filtered_gt = np.full((nq, K_OUT), -1, dtype=np.int32)
match_counts = []
for i in range(nq):
    ql = query_labels[i]
    matching = [nid for nid in gt_ids[i] if node_labels[nid] == ql]
    match_counts.append(len(matching))
    for j, nid in enumerate(matching[:K_OUT]):
        filtered_gt[i, j] = nid

valid = np.sum(filtered_gt[:, 0] >= 0)
avg_matches = np.mean(match_counts)
print(f"  Queries with >= 1 match: {valid}/{nq}")
print(f"  Avg matches in top-{gt_K}: {avg_matches:.1f}")

gt_path = os.path.join(OUTPUT_DIR, "filtered_gt_range.bin")
with open(gt_path, 'wb') as f:
    f.write(struct.pack('II', nq, K_OUT))
    f.write(filtered_gt.tobytes())
print(f"  Saved: {gt_path}")

print("\n=== Done: range labels and GT generated ===")
