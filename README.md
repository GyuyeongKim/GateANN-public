# GateANN: I/O-Efficient Filtered Vector Search on SSDs

This repository contains the artifact for the paper:

> **GateANN: I/O-Efficient Filtered Vector Search on SSDs**

GateANN is an SSD-based graph ANNS system that achieves I/O-efficient filtered vector search on an **unmodified** graph index.
It checks each node's filter metadata **before** issuing SSD I/O and uses **graph tunneling** to traverse non-matching nodes entirely in memory, eliminating up to 90% of SSD reads at 10% selectivity while maintaining recall comparable to post-filtering baselines.

GateANN is implemented as a single additional search mode on top of the [PipeANN](https://github.com/thustorage/PipeANN) codebase.

## Key Results

- Up to **10x fewer SSD I/Os** vs. post-filter baselines
- Up to **7.6x higher throughput** at 10% selectivity with comparable recall
- Works on **any filter predicate** without index rebuild
- Scales from 10M to **1B vectors** on a single commodity server

---

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Software Dependencies](#software-dependencies)
- [Getting Started (~30 min)](#getting-started)
- [Directory Structure](#directory-structure)
- [Datasets](#datasets)
- [Building the Index](#building-the-index)
- [Reproducing Results](#reproducing-results)
  - [Quick Validation (~10 min)](#quick-validation)
  - [Full Reproduction (~24 hours)](#full-reproduction)
  - [Per-Figure Scripts](#per-figure-scripts)
- [Citation](#citation)

---

## Hardware Requirements

| Component | Minimum | Recommended (Paper Setup) |
|-----------|---------|---------------------------|
| CPU | x86-64 with AVX2 | 2× Intel Xeon Silver 4514Y (32 Cores) |
| DRAM | 64 GB | 256 GB |
| SSD | 1 TB NVMe | 2 TB NVMe |
| OS | Linux 5.10+ (io_uring) | Ubuntu 22.04 LTS |

**Notes:**
- `io_uring` support requires Linux kernel >= 5.1 (5.10+ recommended).
- DRAM requirements depend on dataset scale and `R_max` setting:
  - 100M vectors, R_max=32: ~17 GB (index + neighbor store)
  - 1B vectors, R_max=16: ~70 GB
- All experiments in the paper use a single machine (no distributed setup).

## Software Dependencies

```bash
# One-step installation (Ubuntu 22.04)
./scripts/install_deps.sh

# Or manually:
sudo apt-get update
sudo apt-get install -y build-essential cmake g++ libmkl-dev \
    libomp-dev libgoogle-perftools-dev python3 python3-pip
pip3 install matplotlib numpy
cd third_party/liburing && ./configure && make -j && cd ../..
```

**Summary:**
| Dependency | Version | Purpose |
|------------|---------|---------|
| CMake | >= 3.5 | Build system |
| GCC | >= 9.0 | C++17 support |
| Intel MKL / OpenBLAS | any | BLAS for distance computation |
| liburing | >= 2.0 | Async I/O (included) |
| tcmalloc | any | Memory allocator (`libgoogle-perftools-dev`) |
| OpenMP | >= 4.5 | Thread parallelism |
| Python 3 | >= 3.8 | Plotting scripts |
| matplotlib | >= 3.5 | Figure generation |

---

## Getting Started

### 1. Build

```bash
git clone https://github.com/GyuyeongKim/GateANN.git
cd GateANN

# Build liburing (if not installed system-wide)
cd third_party/liburing && ./configure && make -j && cd ../..

# Build GateANN
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 2. Verify Build

After a successful build, the following binaries should exist in `build/tests/`:

| Binary | Description |
|--------|-------------|
| `build_disk_index` | Build Vamana graph index on SSD |
| `build_memory_index` | Build in-memory Vamana index |
| `search_disk_index_fa` | **Main experiment binary** (modes 0/2/8/9) |
| `search_disk_index_yfcc` | YFCC multi-label experiment binary |
| `search_mem_fa` | In-memory Vamana baseline |

### 3. Quick Smoke Test

```bash
# Run a quick validation with BigANN-100M (requires data + index)
./scripts/quick_validate.sh
```

---

## Directory Structure

```
GateANN/
├── README.md                     # This file
├── CMakeLists.txt                # Top-level build configuration
├── include/                      # Header files
│   ├── ssd_index.h               # SSD index with search mode dispatch
│   ├── utils/
│   │   ├── filter_store.h        # [GateANN] Single-label filter store
│   │   ├── spmat_filter_store.h  # [GateANN] Multi-label filter store (CSR)
│   │   └── full_adj_index.h      # [GateANN] Neighbor store for tunneling
│   ├── nbr/                      # PQ distance computation
│   └── filter/                   # Filter selector interface
├── src/                          # Source files
│   ├── search/
│   │   ├── pipe_search.cpp       # Asynchronous pipeline search (core)
│   │   └── beam_search.cpp       # Synchronous beam search (DiskANN)
│   ├── ssd_index.cpp             # SSD index management
│   └── ...
├── tests/                        # Experiment binaries
│   ├── search_disk_index_fa.cpp  # Main filtered search experiments
│   ├── search_disk_index_yfcc.cpp # YFCC multi-label experiments
│   └── search_mem_fa.cpp         # In-memory Vamana baseline
├── scripts/                      # Experiment and plotting scripts
│   ├── run_all.sh                # Master script: reproduce all figures
│   ├── fig01_pareto.sh           # Per-figure experiment scripts
│   ├── plot_pareto_bigann.py     # Per-figure plotting scripts
│   └── ...
├── third_party/
│   └── liburing/                 # io_uring library (included)
├── data/                         # Datasets (download separately)
└── figures/                      # Generated figures (output)
```

### GateANN-Specific Files

The following files contain GateANN's contributions on top of PipeANN:

| File | Lines | Description |
|------|-------|-------------|
| `include/utils/filter_store.h` | ~60 | O(1) single-label pre-IO filter check |
| `include/utils/spmat_filter_store.h` | ~80 | CSR-based multi-label subset predicate |
| `include/utils/full_adj_index.h` | ~140 | Neighbor store for graph tunneling |
| `src/search/pipe_search.cpp` (lines 352-401) | ~50 | Pre-IO filter check + tunneling logic |

---

## Datasets

The paper evaluates on four datasets:

| Dataset | Vectors | Dimensions | Type | Size | Labels |
|---------|---------|------------|------|------|--------|
| BigANN-100M | 100M | 128 | uint8 | 12.8 GB | Synthetic (uniform/Zipf) |
| DEEP-100M | 100M | 96 | float32 | 38.4 GB | Synthetic (uniform) |
| YFCC-10M | 10M | 192 | uint8 | 1.9 GB | Real multi-label |
| BigANN-1B | 1B | 128 | uint8 | 128 GB | Synthetic (uniform) |

### Download

```bash
# BigANN-100M (required for most experiments)
./scripts/setup_data.sh bigann100M

# DEEP-100M
./scripts/setup_data.sh deep100M

# YFCC-10M (includes real multi-label metadata)
./scripts/setup_data.sh yfcc10M

# BigANN-1B (requires ~200 GB disk space for index + data)
./scripts/setup_data.sh bigann1B
```

---

## Building the Index

Parameters: `build_disk_index <type> <base_file> <index_prefix> <R> <L> <PQ_dim> <merge_threshold> <build_threads>`

### BigANN-100M

```bash
./build/tests/build_disk_index uint8 \
    data/bigann100M/bigann100M_base.u8bin \
    data/bigann100M/index/bigann100M \
    128 200 32 80 60
```

### DEEP-100M

```bash
./build/tests/build_disk_index float \
    data/deep100M/deep100M_base.fbin \
    data/deep100M/index/deep100M \
    128 200 32 80 60
```

### YFCC-10M

```bash
./build/tests/build_disk_index uint8 \
    data/yfcc10M/yfcc10M_base.u8bin \
    data/yfcc10M/index/yfcc10M \
    128 200 32 80 60
```

### BigANN-1B

**Note:** Building a 1B-scale index requires substantial resources. The merge phase alone can use ~200 GB of temporary disk space, and DRAM usage peaks at ~128 GB. We used a dedicated 2 TB NVMe SSD for index construction to avoid running out of space.

```bash
./build/tests/build_disk_index uint8 \
    data/bigann1B/bigann1B_base.u8bin \
    data/bigann1B/index/bigann1B \
    128 200 32 80 60
```

### In-memory Vamana Baseline

```bash
./build/tests/build_memory_index uint8 \
    data/bigann100M/bigann100M_base.u8bin \
    data/bigann100M/index/bigann100M_mem \
    128 200
```

### Neighbor Store (FullAdjIndex)

GateANN (mode=8) requires a neighbor store that maps each node to its top-`R_max` graph neighbors in DRAM. The store is built automatically on first run and saved as `<index_prefix>_full_adj_<R_max>.bin`. Subsequent runs load it directly via `mmap`, so the one-time construction cost is amortized across experiments.

### Filtered-DiskANN Comparison (Figure 11)

Figure 11 compares against [Filtered-DiskANN](https://github.com/microsoft/DiskANN), which builds a **label-aware** FilteredVamana graph index. This requires the official DiskANN repository:

```bash
# Clone and build official DiskANN
git clone https://github.com/microsoft/DiskANN.git
cd DiskANN && mkdir build && cd build && cmake .. && make -j$(nproc)
export DISKANN_BUILD_DIR=$(pwd)
cd ../../..
```

Build FilteredVamana indices:

```bash
# BigANN-100M (R=96, FilteredLbuild=100)
$DISKANN_BUILD_DIR/apps/build_disk_index --data_type uint8 --dist_fn l2 \
    --data_path data/bigann100M/bigann100M_base.u8bin \
    --index_path_prefix data/pipeann_indices/bigann100M_official_filtered \
    -R 96 --FilteredLbuild 100 -B 4 -M 80 -T 64 --PQ_disk_bytes 0 \
    --label_file data/filter_exp_100M/bigann100M_node_labels.txt

# DEEP-100M (R=96, FilteredLbuild=128)
$DISKANN_BUILD_DIR/apps/build_disk_index --data_type float --dist_fn l2 \
    --data_path data/deep100M/deep100M_base.fbin \
    --index_path_prefix data/pipeann_indices/deep100M_official_filtered \
    -R 96 --FilteredLbuild 128 -B 4 -M 80 -T 64 --PQ_disk_bytes 0 \
    --label_file data/filter_exp_100M/deep100M_node_labels.txt
```

The `--label_file` expects a text file with one label per line (generated by `setup_data.sh`). The `fig11_fdiskann.sh` script automates both the build (if needed) and search.

---

## Reproducing Results

### Search Modes

| Mode | System | Description |
|------|--------|-------------|
| 0 | DiskANN | Synchronous beam search (post-filter) |
| 2 | PipeANN | Asynchronous pipeline search (post-filter) |
| **8** | **GateANN** | **Pre-IO filter check + graph tunneling** |
| 9 | PipeANN+EarlyFilter | Skip exact distance for non-matching (ablation) |

### Quick Validation

Run a subset of the main experiments to verify the artifact (~10 min):

```bash
./scripts/quick_validate.sh
```

This runs BigANN-100M at 10% selectivity with T=1 and a few L values, comparing DiskANN (mode=0), PipeANN (mode=2), and GateANN (mode=8). Expected output:
- GateANN should achieve **5-8x higher QPS** than PipeANN at comparable recall
- GateANN should use **~10x fewer SSD I/Os** than PipeANN

### Full Reproduction

To reproduce all paper figures (~24 hours total):

```bash
./scripts/run_all.sh
```

This sequentially runs all experiments and generates all figures in `figures/`.

### Per-Figure Scripts

Each figure in the paper has a dedicated experiment + plot script pair:

| Figure | Paper Section | Script | Plot | Est. Time |
|--------|---------------|--------|------|-----------|
| Fig. 1(a-b) | Motivation | `scripts/fig01_motivation.sh` | `scripts/plot_motivation.py` | 30 min |
| Fig. 5(a-d) | Pareto (100M) | `scripts/fig05_pareto_main.sh` | `scripts/plot_pareto_bigann.py`, `plot_pareto_deep.py` | 3 hr |
| Fig. 6 | Thread scaling | `scripts/fig06_thread_scaling.sh` | `scripts/plot_thread_scaling.py` | 30 min |
| Fig. 7(a-b) | I/O reduction | `scripts/fig07_io_reduction.sh` | `scripts/plot_io_reduction.py` | 1 hr |
| Fig. 8(a-b) | BigANN-1B | `scripts/fig08_billion.sh` | `scripts/plot_pareto_bigann1B.py` | 6 hr |
| Fig. 9 | YFCC-10M | `scripts/fig09_yfcc.sh` | `scripts/plot_yfcc_tput.py` | 30 min |
| Fig. 10(a-b) | Vamana comparison | `scripts/fig10_vamana.sh` | `scripts/plot_vamana.py` | 1 hr |
| Fig. 11(a-b) | F-DiskANN comparison | `scripts/fig11_fdiskann.sh` | `scripts/plot_fdiskann.py` | 2 hr |
| Fig. 12 | Selectivity sweep | `scripts/fig12_selectivity.sh` | `scripts/plot_selectivity.py` | 1 hr |
| Fig. 13(a-b) | R_max sweep | `scripts/fig13_nbrs_sweep.sh` | `scripts/plot_nbrs_qps.py`, `plot_nbrs_pareto.py` | 2 hr |
| Fig. 14(a-b) | Zipf distribution | `scripts/fig14_zipf.sh` | `scripts/plot_zipf.py` | 1 hr |
| Fig. 15 | Spatial correlation | `scripts/fig15_spatial.sh` | `scripts/plot_spatial_correlation.py` | 1 hr |
| Fig. 16(a-b) | Range predicates | `scripts/fig16_range.sh` | `scripts/plot_range_predicate.py` | 1 hr |
| Fig. 17(a-b) | Pipeline depth | `scripts/fig17_bw_sweep.sh` | `scripts/plot_bw_sweep.py` | 1 hr |
| Fig. 18(a-b) | Ablation | `scripts/fig18_ablation.sh` | `scripts/plot_early_filter.py` | 1 hr |
| Table 4 | SSD speed impact | `scripts/tab04_ssd_impact.sh` | (printed to stdout) | 30 min |
| Table 5 | Time breakdown | `scripts/tab05_breakdown.sh` | (printed to stdout) | 15 min |

**Usage:**
```bash
# Run a single figure's experiment + plot
./scripts/fig05_pareto_main.sh      # runs experiment, writes results to data/
python3 scripts/plot_pareto_bigann.py   # generates figures/fig_pareto_bigann_{lat,tput}.{eps,png}

# Or use the convenience wrapper
./scripts/run_figure.sh 5           # runs experiment + plot for Figure 5
```

---

## Citation

```bibtex
@misc{gateann,
  title = {{GateANN}: {I/O}-Efficient Filtered Vector Search on {SSDs}},
  author = {Nakyung Lee and Soobin Cho and Jiwoong Park and Gyuyeong Kim},
  year = {2026},
  eprint = {2603.21466},
  archiveprefix = {arXiv},
  primaryclass = {cs.OS},
  url = {https://arxiv.org/abs/2603.21466}
}
```

---

## Acknowledgments

GateANN is built on top of [PipeANN](https://github.com/thustorage/PipeANN). We thank the PipeANN authors for making their codebase available.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
