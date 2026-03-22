#!/usr/bin/env python3
"""
Q5: Generate spatially correlated labels using k-means on BigANN-100M.
- Train k-means (10 clusters) on a subsample
- Assign all 100M vectors to nearest centroid
- Generate labels for each alpha (0.0, 0.5, 1.0):
    alpha=0: purely random
    alpha=0.5: 50% spatial, 50% random
    alpha=1: fully spatial (k-means cluster ID)
- Generate filtered GT from existing unfiltered GT
"""
import struct, os, sys, time
import numpy as np

DATA_DIR = "/home/node33/faiss-dev/benchmarks/data"
BASE_FILE = f"{DATA_DIR}/bigann100M_base.u8bin"
QUERY_FILE = f"{DATA_DIR}/bigann100M_query.u8bin"
GT_FILE = f"{DATA_DIR}/bigann100M_gt.bin"
OUTPUT_DIR = f"{DATA_DIR}/filter_exp_100M/spatial"

NUM_CLUSTERS = 10
ALPHAS = [0.0, 0.5, 1.0]
K_OUT = 10
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(SEED)

###############################################################################
# Step 1: Load vectors (memory-mapped) and train k-means
###############################################################################
print("=== Step 1: k-means on BigANN-100M ===")

with open(BASE_FILE, 'rb') as f:
    npts, dim = struct.unpack('II', f.read(8))
print(f"Base: {npts} pts, {dim} dim")

# Memory-map the vectors
base_fp = open(BASE_FILE, 'rb')
base_vectors = np.ndarray((npts, dim), dtype=np.uint8,
                           buffer=np.memmap(BASE_FILE, dtype=np.uint8,
                                            mode='r', offset=8,
                                            shape=(npts * dim,)))

# Subsample for k-means training
TRAIN_SIZE = 1_000_000
print(f"Sampling {TRAIN_SIZE} vectors for k-means training...")
sample_idx = np.random.choice(npts, TRAIN_SIZE, replace=False)
sample_idx.sort()
train_data = base_vectors[sample_idx].astype(np.float32)

import faiss
print(f"Training k-means with {NUM_CLUSTERS} clusters...")
t0 = time.time()
kmeans = faiss.Kmeans(dim, NUM_CLUSTERS, niter=20, verbose=True, seed=SEED)
kmeans.train(train_data)
del train_data
print(f"k-means training done in {time.time()-t0:.1f}s")

centroids = kmeans.centroids  # shape: (NUM_CLUSTERS, dim)

# Save centroids
np.save(os.path.join(OUTPUT_DIR, "kmeans_centroids.npy"), centroids)

###############################################################################
# Step 2: Assign all 100M vectors to nearest centroid (in chunks)
###############################################################################
print("\n=== Step 2: Cluster assignment ===")
centroid_index = faiss.IndexFlatL2(dim)
centroid_index.add(centroids)

CHUNK = 10_000_000
cluster_labels = np.empty(npts, dtype=np.int32)

t0 = time.time()
for start in range(0, npts, CHUNK):
    end = min(start + CHUNK, npts)
    chunk = base_vectors[start:end].astype(np.float32)
    _, I = centroid_index.search(chunk, 1)
    cluster_labels[start:end] = I.flatten()
    pct = end / npts * 100
    elapsed = time.time() - t0
    print(f"  {end/1e6:.0f}M / {npts/1e6:.0f}M ({pct:.1f}%) - {elapsed:.0f}s")

# Report cluster sizes
for c in range(NUM_CLUSTERS):
    count = np.sum(cluster_labels == c)
    print(f"  Cluster {c}: {count:,} ({count/npts*100:.1f}%)")

np.save(os.path.join(OUTPUT_DIR, "cluster_labels.npy"), cluster_labels)

###############################################################################
# Step 3: Assign queries to nearest centroid
###############################################################################
print("\n=== Step 3: Query cluster assignment ===")
with open(QUERY_FILE, 'rb') as f:
    nq, qdim = struct.unpack('II', f.read(8))
    query_vectors = np.fromfile(f, dtype=np.uint8, count=nq * qdim).reshape(nq, qdim)

print(f"Queries: {nq}, dim={qdim}")
query_float = query_vectors.astype(np.float32)
_, query_clusters = centroid_index.search(query_float, 1)
query_clusters = query_clusters.flatten().astype(np.int32)

###############################################################################
# Step 4: Generate labels and GT for each alpha
###############################################################################
print("\n=== Step 4: Generate labels and filtered GT ===")

# Load unfiltered GT
with open(GT_FILE, 'rb') as f:
    gt_nq, gt_K = struct.unpack('II', f.read(8))
    gt_ids = np.fromfile(f, dtype=np.int32, count=gt_nq * gt_K).reshape(gt_nq, gt_K)
print(f"Unfiltered GT: {gt_nq} queries, K={gt_K}")

for alpha in ALPHAS:
    print(f"\n--- alpha={alpha} ---")
    tag = f"alpha{alpha:.1f}".replace('.', '')

    # Node labels
    if alpha == 0.0:
        node_labels = np.random.randint(0, NUM_CLUSTERS, size=npts, dtype=np.uint8)
    elif alpha == 1.0:
        node_labels = cluster_labels.astype(np.uint8)
    else:
        node_labels = np.empty(npts, dtype=np.uint8)
        mask = np.random.random(npts) < alpha
        node_labels[mask] = cluster_labels[mask].astype(np.uint8)
        node_labels[~mask] = np.random.randint(0, NUM_CLUSTERS, size=np.sum(~mask), dtype=np.uint8)

    # Query labels: same alpha logic
    if alpha == 0.0:
        q_labels = np.random.randint(0, NUM_CLUSTERS, size=nq, dtype=np.uint8)
    elif alpha == 1.0:
        q_labels = query_clusters.astype(np.uint8)
    else:
        q_labels = np.empty(nq, dtype=np.uint8)
        qmask = np.random.random(nq) < alpha
        q_labels[qmask] = query_clusters[qmask].astype(np.uint8)
        q_labels[~qmask] = np.random.randint(0, NUM_CLUSTERS, size=np.sum(~qmask), dtype=np.uint8)

    # Report effective selectivity per query
    sel_per_query = []
    for i in range(nq):
        sel = np.mean(node_labels == q_labels[i])
        sel_per_query.append(sel)
    avg_sel = np.mean(sel_per_query)
    print(f"  Avg selectivity: {avg_sel*100:.2f}%")

    # Save node labels
    nl_path = os.path.join(OUTPUT_DIR, f"node_labels_{tag}.bin")
    with open(nl_path, 'wb') as f:
        f.write(struct.pack('I', npts))
        f.write(node_labels.tobytes())
    print(f"  Saved: {nl_path}")

    # Save query labels
    ql_path = os.path.join(OUTPUT_DIR, f"query_labels_{tag}.bin")
    with open(ql_path, 'wb') as f:
        f.write(struct.pack('I', nq))
        f.write(q_labels.tobytes())
    print(f"  Saved: {ql_path}")

    # Generate filtered GT
    filtered_gt = np.full((nq, K_OUT), -1, dtype=np.int32)
    match_counts = []
    for i in range(nq):
        qlabel = q_labels[i]
        matching = [nid for nid in gt_ids[i] if node_labels[nid] == qlabel]
        match_counts.append(len(matching))
        for j, nid in enumerate(matching[:K_OUT]):
            filtered_gt[i, j] = nid

    valid = np.sum(filtered_gt[:, 0] >= 0)
    avg_matches = np.mean(match_counts)
    print(f"  Queries with >= 1 match: {valid}/{nq}")
    print(f"  Avg matches in top-{gt_K}: {avg_matches:.1f}")

    gt_path = os.path.join(OUTPUT_DIR, f"filtered_gt_{tag}.bin")
    with open(gt_path, 'wb') as f:
        f.write(struct.pack('II', nq, K_OUT))
        f.write(filtered_gt.tobytes())
    print(f"  Saved: {gt_path}")

print("\n=== Done: all labels and GT generated ===")
