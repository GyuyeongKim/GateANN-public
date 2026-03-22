#pragma once

// =============================================================================
// HotNodeCache: frequency-based warm-up caching for SSD graph ANNS.
// =============================================================================
// Key design: two-phase warmup -> finalize. After finalization the cache is
// read-only and lock-free. Used by:
//   - mode=4 (CACHED_PIPE_SEARCH):     lazy freq cache
//   - mode=5 (BFS_CACHED_PIPE_SEARCH): BFS static cache (DiskANN-style variant)
// =============================================================================

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace pipeann {

// Hot Node Cache for workload-aware disk ANN search.
//
// Caches frequently accessed graph nodes (full DiskNode data: vector + neighbor list)
// in memory to avoid redundant disk I/O under skewed query distributions.
//
// Two-phase operation:
//   Phase 1 (Warmup): Run queries through normal pipe_search. Each disk-fetched node
//     is recorded (ID + frequency + data copy). Thread-safe via mutex.
//   Phase 2 (Serving): Cache is finalized (top-K nodes kept). Lookups are lock-free
//     (read-only after finalization).
template<typename T>
class HotNodeCache {
 public:
  HotNodeCache(size_t max_nodes, size_t max_node_len)
      : max_nodes_(max_nodes), max_node_len_(max_node_len), populated_(false) {}

  ~HotNodeCache() {
    for (auto &[id, buf] : cache_) {
      free(buf);
    }
  }

  // --- Warmup phase ---

  // Record a node access and store its data. Thread-safe.
  // node_data points to the start of the node's coords in the page buffer.
  void record_and_store(uint32_t id, const char *node_data) {
    std::lock_guard<std::mutex> lock(mu_);
    freq_[id]++;
    if (cache_.find(id) == cache_.end()) {
      char *buf = (char *)malloc(max_node_len_);
      memcpy(buf, node_data, max_node_len_);
      cache_[id] = buf;
    }
  }

  // Finalize: keep only the top max_nodes_ most frequently accessed nodes.
  // After this, the cache is read-only (serving mode).
  void finalize() {
    std::vector<std::pair<uint32_t, uint64_t>> sorted_freq(freq_.begin(), freq_.end());
    std::sort(sorted_freq.begin(), sorted_freq.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });

    // Determine which nodes to keep
    std::unordered_map<uint32_t, char *> new_cache;
    size_t keep_count = std::min(max_nodes_, sorted_freq.size());
    for (size_t i = 0; i < keep_count; i++) {
      uint32_t id = sorted_freq[i].first;
      auto it = cache_.find(id);
      if (it != cache_.end()) {
        new_cache[id] = it->second;
        cache_.erase(it);
      }
    }

    // Free non-kept nodes
    for (auto &[id, buf] : cache_) {
      free(buf);
    }
    cache_ = std::move(new_cache);

    // Store warmup stats before clearing
    total_warmup_accesses_ = 0;
    for (auto &[id, f] : freq_) {
      total_warmup_accesses_ += f;
    }
    // Sum frequencies of cached nodes → estimated cache hit rate
    cached_freq_sum_ = 0;
    for (size_t i = 0; i < keep_count; i++) {
      cached_freq_sum_ += sorted_freq[i].second;
    }
    unique_nodes_seen_ = freq_.size();
    if (!sorted_freq.empty()) {
      max_freq_ = sorted_freq[0].second;
      min_freq_ = sorted_freq.back().second;
      if (keep_count > 0) {
        cached_min_freq_ = sorted_freq[keep_count - 1].second;
      }
    }

    // Retain sorted_freq for later resize (auto-tuning cache size post-warmup)
    sorted_freq_ = std::move(sorted_freq);

    freq_.clear();
    populated_ = true;
  }

  // Resize the cache to a new target K, using stored warmup frequency distribution.
  // Must be called after finalize(). Evicts low-frequency nodes or adds high-frequency ones.
  // Returns the actual new size (may differ if fewer nodes were seen during warmup).
  size_t resize(size_t new_max_nodes) {
    if (sorted_freq_.empty()) return cache_.size();  // no warmup data

    size_t new_keep = std::min(new_max_nodes, sorted_freq_.size());

    // Build new cache from sorted_freq_
    std::unordered_map<uint32_t, char *> new_cache;
    for (size_t i = 0; i < new_keep; i++) {
      uint32_t id = sorted_freq_[i].first;
      // Check if already in cache
      auto it = cache_.find(id);
      if (it != cache_.end()) {
        new_cache[id] = it->second;
        cache_.erase(it);
      }
      // If not in cache: we don't have the data anymore (warmup freed non-top-K nodes).
      // Can only grow within already-cached nodes or shrink.
    }

    // Free nodes no longer kept
    for (auto &[id, buf] : cache_) {
      free(buf);
    }
    cache_ = std::move(new_cache);

    // Update cached stats
    max_nodes_ = new_max_nodes;
    cached_freq_sum_ = 0;
    for (size_t i = 0; i < new_keep && i < sorted_freq_.size(); i++) {
      cached_freq_sum_ += sorted_freq_[i].second;
    }
    if (new_keep > 0 && new_keep <= sorted_freq_.size()) {
      cached_min_freq_ = sorted_freq_[new_keep - 1].second;
    }

    return cache_.size();
  }

  // Compute the optimal cache size such that estimated hit_rate ≈ target_hit_rate.
  // Uses binary search over the sorted frequency distribution.
  // target_hit_rate: fraction of accesses to serve from cache (e.g., L/(L+4)).
  size_t optimal_k_for_hit_rate(double target_hit_rate) const {
    if (sorted_freq_.empty() || total_warmup_accesses_ == 0) return max_nodes_;
    double target_freq_sum = target_hit_rate * total_warmup_accesses_;

    // Binary search: find K such that sum(freq[0..K-1]) ≈ target_freq_sum
    double cumsum = 0;
    for (size_t k = 0; k < sorted_freq_.size(); k++) {
      cumsum += sorted_freq_[k].second;
      if (cumsum >= target_freq_sum) {
        return k + 1;
      }
    }
    return sorted_freq_.size();  // all nodes needed
  }

  // --- Direct store (for BFS cache, no frequency tracking) ---

  void direct_store(uint32_t id, const char *node_data) {
    if (cache_.size() >= max_nodes_) return;
    if (cache_.find(id) != cache_.end()) return;
    char *buf = (char *)malloc(max_node_len_);
    memcpy(buf, node_data, max_node_len_);
    cache_[id] = buf;
  }

  // Mark cache as ready for serving (used by BFS mode, skips finalize)
  void mark_populated() { populated_ = true; }

  // --- Serving phase ---

  bool is_populated() const { return populated_; }
  bool is_warming() const { return !populated_ && max_nodes_ > 0; }

  bool lookup(uint32_t id) const {
    return cache_.find(id) != cache_.end();
  }

  char *get_node_buf(uint32_t id) {
    auto it = cache_.find(id);
    return (it != cache_.end()) ? it->second : nullptr;
  }

  // Get all cached node IDs (for building exclusion sets in hybrid cache)
  std::unordered_set<uint32_t> get_cached_ids() const {
    std::unordered_set<uint32_t> ids;
    for (auto &[id, buf] : cache_) ids.insert(id);
    return ids;
  }

  size_t size() const { return cache_.size(); }
  size_t max_nodes() const { return max_nodes_; }
  size_t max_node_len() const { return max_node_len_; }

  // Warmup diagnostics (valid after finalize)
  uint64_t total_warmup_accesses() const { return total_warmup_accesses_; }
  size_t unique_nodes_seen() const { return unique_nodes_seen_; }
  uint64_t max_freq() const { return max_freq_; }
  uint64_t min_freq() const { return min_freq_; }
  uint64_t cached_min_freq() const { return cached_min_freq_; }
  uint64_t cached_freq_sum() const { return cached_freq_sum_; }

  // Estimated fraction of accesses that hit the cache (from warmup)
  double estimated_hit_rate() const {
    return total_warmup_accesses_ > 0
               ? (double)cached_freq_sum_ / total_warmup_accesses_
               : 0.0;
  }

 private:
  std::unordered_map<uint32_t, uint64_t> freq_;   // node ID -> access count (warmup)
  std::unordered_map<uint32_t, char *> cache_;     // node ID -> cached node data
  size_t max_nodes_;       // max nodes to cache
  size_t max_node_len_;    // bytes per node (fixed)
  bool populated_;         // true after finalize()
  std::mutex mu_;          // protects freq_ and cache_ during warmup

  // Warmup diagnostics
  uint64_t total_warmup_accesses_ = 0;
  size_t unique_nodes_seen_ = 0;
  uint64_t max_freq_ = 0;
  uint64_t min_freq_ = 0;
  uint64_t cached_min_freq_ = 0;
  uint64_t cached_freq_sum_ = 0;

  // Retained for post-warmup cache resize (auto-tuning). Sorted descending by freq.
  std::vector<std::pair<uint32_t, uint64_t>> sorted_freq_;
};

}  // namespace pipeann
