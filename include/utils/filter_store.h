#pragma once

// =============================================================================
// PIPEANN ORIGINAL CONTRIBUTION: FilterStore for Filter-IO Decoupling
// =============================================================================
// Stores per-node uint8 labels in DRAM for O(1) pre-IO filter checking.
// Key idea: check filter BEFORE sending SSD IO, not after.
//   - Filter-pass nodes: send IO normally
//   - Filter-fail nodes: skip IO, tunnel via FullAdjIndex for connectivity
//
// Cost: 1 byte/node. For 10M nodes = 10MB (negligible).
// =============================================================================

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pipeann {

class FilterStore {
 public:
  // Load from binary file: [uint32_t n_nodes][uint8_t label[0..n-1]]
  void load(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("FilterStore: cannot open " + path);
    uint32_t n;
    in.read((char *)&n, sizeof(n));
    labels_.resize(n);
    in.read((char *)labels_.data(), n);
    if (!in) throw std::runtime_error("FilterStore: read failed " + path);
  }

  void save(const std::string &path) const {
    std::ofstream out(path, std::ios::binary);
    uint32_t n = (uint32_t)labels_.size();
    out.write((char *)&n, sizeof(n));
    out.write((char *)labels_.data(), n);
  }

  void resize_and_set(uint32_t n, uint8_t default_label = 0) {
    labels_.assign(n, default_label);
  }
  void set_label(uint32_t id, uint8_t label) { labels_[id] = label; }

  // Returns true if node PASSES filter (should be IO'd and included in result).
  bool passes(uint32_t node_id, uint8_t query_label) const {
    if (node_id >= labels_.size()) return true;
    return labels_[node_id] == query_label;
  }

  bool loaded() const { return !labels_.empty(); }
  size_t size() const { return labels_.size(); }
  uint8_t get_label(uint32_t id) const { return labels_[id]; }

 private:
  std::vector<uint8_t> labels_;
};

}  // namespace pipeann
