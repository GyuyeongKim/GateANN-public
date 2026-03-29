#pragma once

// =============================================================================
// GateANN: FullAdjIndex for Filter Tunneling
// =============================================================================
// Stores compact neighbor lists for ALL nodes in DRAM.
// Used when a filter-fail node needs neighbor expansion WITHOUT an SSD IO.
// ("Tunneling": traverse through invalid nodes to reach valid ones for free.)
//
// Memory: n_nodes × (1 + max_nbrs) × 4 bytes
//   SIFT1M,  max_nbrs=32: 1M × 33 × 4 =  132MB
//   deep10M, max_nbrs=32: 10M × 33 × 4 = 1.3GB
//
// Built by sequentially reading the disk index once at startup.
//
// Supports two allocation modes:
//   1. Populate (default): all pages resident in DRAM (current behavior)
//   2. Lazy mmap: file-backed mmap without MAP_POPULATE; pages faulted on demand
// =============================================================================

#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>

#include "utils/log.h"

namespace pipeann {

class FullAdjIndex {
 public:
  FullAdjIndex() = default;
  ~FullAdjIndex() { cleanup(); }

  // Disable copy (mmap resource management)
  FullAdjIndex(const FullAdjIndex &) = delete;
  FullAdjIndex &operator=(const FullAdjIndex &) = delete;

  // Build mode: allocate in memory via std::vector
  void init(size_t n_nodes, uint32_t max_nbrs) {
    cleanup();
    n_nodes_  = n_nodes;
    max_nbrs_ = max_nbrs;
    stride_   = 1 + max_nbrs;         // [nnbrs | nbr[0] | nbr[1] | ...]
    data_vec_.assign(n_nodes * stride_, 0);
    data_ = data_vec_.data();
  }

  void set_adj(uint32_t id, uint32_t nnbrs, const uint32_t *nbrs) {
    uint32_t *ptr = const_cast<uint32_t *>(data_) + (size_t)id * stride_;
    uint32_t keep = std::min(nnbrs, max_nbrs_);
    ptr[0] = keep;
    memcpy(ptr + 1, nbrs, keep * sizeof(uint32_t));
  }

  void get_adj(uint32_t id, uint32_t &nnbrs, const uint32_t *&nbrs) const {
    const uint32_t *ptr = data_ + (size_t)id * stride_;
    nnbrs = ptr[0];
    nbrs  = ptr + 1;
  }

  // ---- Persistence: save to file / load via mmap ----

  // File format: [uint64 n_nodes][uint64 max_nbrs][uint64 stride][uint32 data...]
  bool save(const std::string &path) const {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) {
      LOG(ERROR) << "FullAdjIndex::save: cannot open " << path;
      return false;
    }
    uint64_t header[3] = {(uint64_t)n_nodes_, (uint64_t)max_nbrs_, (uint64_t)stride_};
    fwrite(header, sizeof(uint64_t), 3, f);
    fwrite(data_, sizeof(uint32_t), n_nodes_ * stride_, f);
    fclose(f);
    LOG(INFO) << "FullAdjIndex saved: " << path
              << " (" << memory_gb() << " GB)";
    return true;
  }

  // Load from file via mmap.  populate=true → MAP_POPULATE (pre-fault all pages).
  bool load_mmap(const std::string &path, bool populate = true) {
    cleanup();
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      LOG(ERROR) << "FullAdjIndex::load_mmap: cannot open " << path;
      return false;
    }

    // Read header
    uint64_t header[3];
    ssize_t hread = ::read(fd, header, sizeof(header));
    if (hread != (ssize_t)sizeof(header)) {
      close(fd);
      return false;
    }
    n_nodes_  = (size_t)header[0];
    max_nbrs_ = (uint32_t)header[1];
    stride_   = (size_t)header[2];

    // File size
    struct stat st;
    fstat(fd, &st);
    mmap_size_ = (size_t)st.st_size;

    int flags = MAP_PRIVATE;
    if (populate) flags |= MAP_POPULATE;

    mmap_base_ = mmap(nullptr, mmap_size_, PROT_READ, flags, fd, 0);
    close(fd);

    if (mmap_base_ == MAP_FAILED) {
      LOG(ERROR) << "FullAdjIndex::load_mmap: mmap failed";
      mmap_base_ = nullptr;
      return false;
    }

    // Data starts after header (3 × uint64_t = 24 bytes)
    data_ = reinterpret_cast<const uint32_t *>(
        static_cast<const char *>(mmap_base_) + sizeof(header));
    is_mmap_ = true;

    // Free the build vector if any
    data_vec_.clear();
    data_vec_.shrink_to_fit();

    LOG(INFO) << "FullAdjIndex loaded via mmap"
              << (populate ? " (MAP_POPULATE)" : " (lazy)")
              << ": " << n_nodes_ << " nodes, " << memory_gb() << " GB";
    return true;
  }

  // Switch from in-memory vector to file-backed mmap (saves vector memory)
  bool persist_and_mmap(const std::string &path, bool populate = true) {
    if (!save(path)) return false;
    return load_mmap(path, populate);
  }

  bool loaded()   const { return data_ != nullptr; }
  size_t size()   const { return n_nodes_; }
  uint32_t max_nbrs() const { return max_nbrs_; }

  double memory_gb() const {
    if (is_mmap_) {
      return mmap_size_ / (1024.0 * 1024.0 * 1024.0);
    }
    return data_vec_.size() * sizeof(uint32_t) / (1024.0 * 1024.0 * 1024.0);
  }

 private:
  void cleanup() {
    if (is_mmap_ && mmap_base_) {
      munmap(mmap_base_, mmap_size_);
      mmap_base_ = nullptr;
    }
    data_vec_.clear();
    data_vec_.shrink_to_fit();
    data_ = nullptr;
    is_mmap_ = false;
  }

  size_t   n_nodes_  = 0;
  uint32_t max_nbrs_ = 0;
  size_t   stride_   = 0;

  // Build phase: std::vector storage
  std::vector<uint32_t> data_vec_;

  // Access pointer (points to either data_vec_.data() or mmap'd region)
  const uint32_t *data_ = nullptr;

  // mmap state
  bool    is_mmap_   = false;
  void   *mmap_base_ = nullptr;
  size_t  mmap_size_ = 0;
};

}  // namespace pipeann
