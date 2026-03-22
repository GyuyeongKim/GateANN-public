#pragma once

// =============================================================================
// SpmatFilterStore: Multi-label filter store for YFCC-style metadata
// =============================================================================
// Stores per-node label sets in DRAM using CSR format.
// Supports subset predicate: query labels ⊆ node labels.
// Used for pre-I/O filter checking in mode=8 with multi-label datasets.
//
// Filter data format (query side):
//   [uint32_t count][uint32_t label1]...[uint32_t labelN]
// =============================================================================

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils/log.h"

namespace pipeann {

class SpmatFilterStore {
 public:
  // Load from spmat file (CSR format):
  //   header: 3 x int64 (nrow, ncol, nnz)
  //   indptr: (nrow+1) x int64
  //   indices: nnz x int32
  //   data: nnz x float32
  void load(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("SpmatFilterStore: cannot open " + path);

    int64_t nrow, ncol, nnz;
    in.read((char *)&nrow, sizeof(int64_t));
    in.read((char *)&ncol, sizeof(int64_t));
    in.read((char *)&nnz, sizeof(int64_t));

    LOG(INFO) << "SpmatFilterStore: loading nrow=" << nrow
              << " ncol=" << ncol << " nnz=" << nnz;

    // Read indptr
    std::vector<int64_t> indptr(nrow + 1);
    in.read((char *)indptr.data(), (nrow + 1) * sizeof(int64_t));

    // Read indices
    std::vector<int32_t> indices(nnz);
    in.read((char *)indices.data(), nnz * sizeof(int32_t));

    // Read data
    std::vector<float> data(nnz);
    in.read((char *)data.data(), nnz * sizeof(float));
    in.close();

    // Build CSR: flatten into indptr_ (int64) and labels_ (uint32)
    n_nodes_ = (uint32_t)nrow;
    indptr_.resize(nrow + 1);

    // First pass: count non-zero entries per row
    std::vector<uint32_t> counts(nrow, 0);
    for (int64_t i = 0; i < nrow; i++) {
      for (int64_t j = indptr[i]; j < indptr[i + 1]; j++) {
        if (data[j] != 0.0f) counts[i]++;
      }
    }

    // Build new indptr
    indptr_[0] = 0;
    for (int64_t i = 0; i < nrow; i++) {
      indptr_[i + 1] = indptr_[i] + counts[i];
    }
    total_labels_ = indptr_[nrow];

    // Allocate and fill labels (sorted per node for faster subset check)
    labels_.resize(total_labels_);
    std::vector<int64_t> pos(nrow);
    for (int64_t i = 0; i < nrow; i++) pos[i] = indptr_[i];

    for (int64_t i = 0; i < nrow; i++) {
      for (int64_t j = indptr[i]; j < indptr[i + 1]; j++) {
        if (data[j] != 0.0f) {
          labels_[pos[i]++] = (uint32_t)indices[j];
        }
      }
      // Sort labels for this node
      std::sort(labels_.begin() + indptr_[i],
                labels_.begin() + indptr_[i + 1]);
    }

    double mem_mb = (indptr_.size() * 8 + labels_.size() * 4) / (1024.0 * 1024.0);
    LOG(INFO) << "SpmatFilterStore loaded: " << n_nodes_ << " nodes, "
              << total_labels_ << " total labels, "
              << std::fixed << std::setprecision(1) << mem_mb << " MB";
  }

  // Check if query labels are a subset of node's labels.
  // filter_data points to: [uint32_t count][uint32_t label1]...[uint32_t labelN]
  // (query labels must be sorted)
  bool passes(uint32_t node_id, const void *filter_data) const {
    if (node_id >= n_nodes_) return true;
    if (!filter_data) return true;

    uint32_t q_count;
    memcpy(&q_count, filter_data, sizeof(uint32_t));
    if (q_count == 0) return true;

    const uint32_t *q_labels = (const uint32_t *)((const char *)filter_data + sizeof(uint32_t));

    // Node's labels (sorted)
    int64_t n_start = indptr_[node_id];
    int64_t n_end = indptr_[node_id + 1];
    uint32_t n_count = (uint32_t)(n_end - n_start);

    if (n_count == 0) return false;

    // Subset check: every query label must exist in node's label set
    // Both are sorted, so use merge-like scan
    uint32_t qi = 0, ni = 0;
    while (qi < q_count && ni < n_count) {
      uint32_t ql = q_labels[qi];
      uint32_t nl = labels_[n_start + ni];
      if (ql == nl) {
        qi++;
        ni++;
      } else if (nl < ql) {
        ni++;
      } else {
        return false;  // query label not found in node
      }
    }
    return qi == q_count;
  }

  bool loaded() const { return n_nodes_ > 0; }
  size_t size() const { return n_nodes_; }

 private:
  uint32_t n_nodes_ = 0;
  int64_t total_labels_ = 0;
  std::vector<int64_t> indptr_;     // n_nodes+1 entries
  std::vector<uint32_t> labels_;    // flattened sorted labels
};

}  // namespace pipeann
