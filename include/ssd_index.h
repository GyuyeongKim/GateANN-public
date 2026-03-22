#pragma once
#include <immintrin.h>
#include <cassert>
#include <cstdint>
#include <queue>
#include <string>
#include <set>
#include <unordered_map>
#include <random>
#include <omp.h>

#include "aligned_file_reader.h"
#include "ssd_index_defs.h"
#include "filter/selector.h"
#include "utils/concurrent_queue.h"
#include "utils/lock_table.h"
#include "utils/percentile_stats.h"
#include "nbr/nbr.h"
#include "utils.h"
#include "index.h"
#include "utils/hot_node_cache.h"
#include "utils/filter_store.h"
#include "utils/spmat_filter_store.h"
#include "utils/full_adj_index.h"

enum SearchMode {
  BEAM_SEARCH = 0, PAGE_SEARCH = 1, PIPE_SEARCH = 2, CORO_SEARCH = 3,
  CACHED_PIPE_SEARCH = 4, BFS_CACHED_PIPE_SEARCH = 5,
  CACHED_BEAM_SEARCH = 6, ADJ_CACHED_PIPE_SEARCH = 7,
  // PIPEANN ORIGINAL: Filter-aware pipe search (pre-IO filter check + tunneling)
  FILTER_AWARE_PIPE_SEARCH = 8,
  EARLY_FILTER_PIPE_SEARCH = 9,
};

namespace pipeann {
  template<typename T, typename TagT = uint32_t>
  class SSDIndex {
   public:
    static constexpr uint32_t kAllocatedID = std::numeric_limits<uint32_t>::max() - 1;

    std::unique_ptr<Index<T, uint32_t>> mem_index_;  // in-memory navigation graph

    // Index metadata (consolidated).
    SSDIndexMetadata<T> meta_;

    // For updates:
    // meta.npoints is the index's initial size (constant between two merges).
    // cur_id is the ID to be allocated (+1 for each insert, starting from meta.npoints)
    // cur_loc is the tail of the index file, which will be greater than cur_id if overprovisioned.
    // - This is because holes may exist in the index for update combining.
    std::atomic<uint64_t> cur_id, cur_loc;

    SSDIndex(pipeann::Metric m, std::shared_ptr<AlignedFileReader> &file_reader,
             AbstractNeighbor<T> *nbr = new PQNeighbor<T>(), bool tags = false,
             IndexBuildParameters *parameters = nullptr);

    ~SSDIndex();

    // Use node_from_page() to create instances for DiskNode.
    // Params: location (id2loc first if using ID), the page-aligned buffer.
    // The offset is calculated in the constructor.
    inline DiskNode<T> node_from_page(char *page_buf, uint32_t loc) {
      return DiskNode<T>(page_buf, loc, meta_);
    }

    // Size of the data region in a DiskNode.
    inline uint64_t node_label_size() const {
      return meta_.label_size;
    }

    // Unaligned offset to location.
    inline uint64_t u_loc_offset(uint64_t loc) {
      return loc * meta_.max_node_len;  // compacted store.
    }

    inline uint64_t u_loc_offset_nbr(uint64_t loc) {
      return loc * meta_.max_node_len + meta_.data_dim * sizeof(T);
    }

    // Avoid integer overflow when * SECTOR_LEN.
    inline uint64_t loc_sector_no(uint64_t loc) {
      return 1 + (meta_.nnodes_per_sector > 0 ? loc / meta_.nnodes_per_sector
                                              : loc * DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN));
    }

    inline uint64_t sector_to_loc(uint64_t sector_no, uint32_t sector_off) {
      return meta_.nnodes_per_sector == 0
                 ? (sector_no - 1) / DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN)  // sector_off == 0.
                 : (sector_no - 1) * meta_.nnodes_per_sector + sector_off;
    }

    void init_metadata(const SSDIndexMetadata<T> &meta) {
      meta.print();
      this->meta_ = meta;
      this->cur_id = this->cur_loc = meta.npoints;
      this->aligned_dim = ROUND_UP(meta_.data_dim, 8);
      this->params.R = meta.range;
      this->size_per_io = SECTOR_LEN * (meta_.nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN));
      LOG(INFO) << "Size per IO: " << size_per_io;

      // Aligned.
      if (meta_.nnodes_per_sector != 0 && meta_.npoints % meta_.nnodes_per_sector != 0) {
        cur_loc += meta_.nnodes_per_sector - (meta_.npoints % meta_.nnodes_per_sector);
      }
      LOG(INFO) << "Cur location: " << this->cur_loc;

      // Update-related metadata, if not initialized in constructor, initialize here.
      if (params.L == 0) {
        // Experience values.
        LOG(INFO) << "Automatically set the update-related parameters.";
        params.set(meta.range, meta.range + 32, 384, 1.2, 0, true, 4);
        params.print();
      }
    }

    void init_query_buf(QueryBuffer<T> &buf) {
      pipeann::alloc_aligned((void **) &buf.coord_scratch, this->aligned_dim * sizeof(T), 8 * sizeof(T));
      pipeann::alloc_aligned((void **) &buf.sector_scratch, MAX_N_SECTOR_READS * size_per_io, SECTOR_LEN);
      pipeann::alloc_aligned((void **) &buf.nbr_vec_scratch,
                             MAX_N_EDGES * AbstractNeighbor<T>::MAX_BYTES_PER_NBR * sizeof(uint8_t), 256);
      pipeann::alloc_aligned((void **) &buf.nbr_ctx_scratch, ROUND_UP(nbr_handler->query_ctx_size(), 256), 256);
      pipeann::alloc_aligned((void **) &buf.aligned_dist_scratch, MAX_N_EDGES * sizeof(float), 256);
      pipeann::alloc_aligned((void **) &buf.aligned_query_T, this->aligned_dim * sizeof(T), 8 * sizeof(T));

      buf.visited = new tsl::robin_set<uint64_t>(4096);
      buf.page_visited = new tsl::robin_set<unsigned>(4096);

      memset(buf.sector_scratch, 0, MAX_N_SECTOR_READS * SECTOR_LEN);
      memset(buf.coord_scratch, 0, this->aligned_dim * sizeof(T));
      memset(buf.aligned_query_T, 0, this->aligned_dim * sizeof(T));
    }

    QueryBuffer<T> *pop_query_buf(const T *query) {
      QueryBuffer<T> *data = this->thread_data_queue.pop();
      while (data == this->thread_data_queue.null_T) {
        this->thread_data_queue.wait_for_push_notify();
        data = this->thread_data_queue.pop();
      }

      if (likely(query != nullptr)) {
        if (this->metric == Metric::COSINE) {
          // Data has been normalized. Normalize search vector too.
          pipeann::normalize_data(data->aligned_query_T, query, meta_.data_dim);
        } else {
          memcpy(data->aligned_query_T, query, meta_.data_dim * sizeof(T));
        }
      }
      return data;
    }

    void push_query_buf(QueryBuffer<T> *data) {
      this->thread_data_queue.push(data);
      this->thread_data_queue.push_notify_all();
    }

    // Load compressed data, and obtains the handle to the disk-resident index.
    int load(const char *index_prefix, uint32_t num_threads, bool use_page_search = false);

    void load_mem_index(const std::string &mem_index_path);

    // Load disk index to memory index.
    Index<T, TagT> *load_to_mem(const std::string &filename);

    // Search supporting update.
    size_t beam_search(const T *query, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT *res_tags, float *res_dists, const uint64_t beam_width, QueryStats *stats = nullptr,
                       tsl::robin_set<uint32_t> *deleted_nodes = nullptr, bool dyn_search_l = true,
                       const void *filter_data = nullptr);

    size_t coro_search(T **queries, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT **res_tags, float **res_dists, const uint64_t beam_width, int N);

    // Read-only search algorithms.
    size_t page_search(const T *query, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT *res_tags, float *res_dists, const uint64_t beam_width, QueryStats *stats = nullptr);

    size_t pipe_search(const T *query, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT *res_tags, float *res_dists, const uint64_t beam_width, QueryStats *stats = nullptr,
                       AbstractSelector *selector = nullptr, const void *filter_data = nullptr,
                       const uint64_t relaxed_monotonicity_l = 0);

    // Hot node cache for workload-aware search
    void init_cache(size_t max_nodes) {
      hot_cache_ = std::make_unique<HotNodeCache<T>>(max_nodes, meta_.max_node_len);
    }

    void finalize_cache() {
      if (hot_cache_) hot_cache_->finalize();
    }

    HotNodeCache<T> *get_cache() const { return hot_cache_.get(); }

    size_t get_max_node_len() const { return meta_.max_node_len; }
    uint32_t get_range() const { return meta_.range; }

    // Build BFS-based static cache: cache nodes closest to medoid in hop distance.
    // This is the DiskANN-style c-hop BFS cache (deterministic, query-independent).
    void build_bfs_cache(size_t max_nodes) {
      hot_cache_ = std::make_unique<HotNodeCache<T>>(max_nodes, meta_.max_node_len);

      void *ctx = reader->get_ctx();

      // Allocate aligned buffer for sector reads
      char *sector_buf;
      pipeann::alloc_aligned((void **)&sector_buf, size_per_io, SECTOR_LEN);

      // BFS from entry point (medoid)
      std::queue<uint32_t> bfs_queue;
      tsl::robin_set<uint32_t> visited;

      uint32_t start_id = meta_.entry_point_id;
      bfs_queue.push(start_id);
      visited.insert(start_id);

      size_t cached_count = 0;
      size_t n_reads = 0;

      while (!bfs_queue.empty() && cached_count < max_nodes) {
        uint32_t id = bfs_queue.front();
        bfs_queue.pop();

        // Read the sector containing this node
        uint32_t loc = id2loc(id);
        uint64_t sector_no = loc_sector_no(loc);

        IORequest req(sector_no * SECTOR_LEN, size_per_io, sector_buf,
                      u_loc_offset(loc), meta_.max_node_len, sector_buf);
        std::vector<IORequest> reqs = {req};
        reader->read(reqs, ctx, false);
        n_reads++;

        // Parse node and store in cache
        DiskNode<T> node = node_from_page(sector_buf, loc);
        hot_cache_->direct_store(id, (char *)node.coords);
        cached_count++;

        // Enqueue unvisited neighbors (BFS order = hop distance order)
        for (uint32_t i = 0; i < node.nnbrs; i++) {
          uint32_t nbr_id = node.nbrs[i];
          if (visited.find(nbr_id) == visited.end()) {
            visited.insert(nbr_id);
            bfs_queue.push(nbr_id);
          }
        }
      }

      hot_cache_->mark_populated();
      pipeann::aligned_free(sector_buf);

      LOG(INFO) << "BFS cache built: " << cached_count << " nodes, "
                << n_reads << " disk reads, "
                << (cached_count * meta_.max_node_len / (1024.0 * 1024.0)) << " MB";
    }

    int insert_in_place(const T *point, const TagT &tag, tsl::robin_set<uint32_t> *deletion_set = nullptr);

    // Merge deletes (NOTE: index read-only during merge.)
    void merge_deletes(const std::string &in_path_prefix, const std::string &out_path_prefix,
                       const std::vector<TagT> &deleted_nodes, const tsl::robin_set<TagT> &deleted_nodes_set,
                       uint32_t nthreads, const uint32_t &n_sampled_nbrs);

    // After merge, reload the index.
    void reload(const char *index_prefix, uint32_t num_threads);

    void write_metadata_and_pq(const std::string &in_path_prefix, const std::string &out_path_prefix,
                               std::vector<TagT> *new_tags = nullptr);

    void copy_index(const std::string &prefix_in, const std::string &prefix_out);

    // =========================================================================
    // PIPEANN ORIGINAL CONTRIBUTION — Filter-Aware IO (mode=8)
    // =========================================================================
    void load_filter_store(const std::string &path) {
      filter_store_ = std::make_unique<FilterStore>();
      filter_store_->load(path);
      LOG(INFO) << "FilterStore loaded: " << filter_store_->size() << " nodes";
    }

    // build_full_adj: build or load the FullAdjIndex.
    //   lazy_adj: false (default) = MAP_POPULATE; true = lazy mmap
    void build_full_adj(uint32_t max_nbrs = 32, bool lazy_adj = false) {
      // Construct file paths from the disk index path
      std::string base = disk_index_file;
      // Strip trailing "_disk.index" if present to get prefix
      auto pos = base.rfind("_disk.index");
      if (pos != std::string::npos) base = base.substr(0, pos);
      std::string adj_path = base + ".full_adj.R" + std::to_string(max_nbrs);

      // Try loading existing file
      {
        struct stat st;
        if (stat(adj_path.c_str(), &st) == 0) {
          LOG(INFO) << "Loading FullAdjIndex from " << adj_path;
          full_adj_ = std::make_unique<FullAdjIndex>();
          if (full_adj_->load_mmap(adj_path, !lazy_adj)) return;
          full_adj_.reset();
        }
      }

      // Build from disk index (sequential scan)
      LOG(INFO) << "Building FullAdjIndex (max_nbrs=" << max_nbrs << ")...";
      full_adj_ = std::make_unique<FullAdjIndex>();
      full_adj_->init(meta_.npoints, max_nbrs);

      void *ctx = reader->get_ctx();
      char *sector_buf;
      pipeann::alloc_aligned((void **)&sector_buf, size_per_io, SECTOR_LEN);

      size_t n_sectors = 1 + (meta_.nnodes_per_sector > 0
                                  ? DIV_ROUND_UP(meta_.npoints, meta_.nnodes_per_sector)
                                  : meta_.npoints * DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN));

      for (size_t sec = 1; sec < n_sectors; sec++) {
        IORequest req(sec * SECTOR_LEN, size_per_io, sector_buf, 0, 0, sector_buf);
        std::vector<IORequest> reqs = {req};
        reader->read(reqs, ctx, false);

        if (meta_.nnodes_per_sector > 0) {
          for (uint32_t off = 0; off < meta_.nnodes_per_sector; off++) {
            uint32_t loc = (sec - 1) * meta_.nnodes_per_sector + off;
            if (loc >= meta_.npoints) break;
            uint32_t id = loc;
            if (!loc2id_.empty() && loc < loc2id_.size()) id = loc2id_[loc];
            if (id >= meta_.npoints) continue;
            DiskNode<T> node = node_from_page(sector_buf, off);
            full_adj_->set_adj(id, node.nnbrs, node.nbrs);
          }
        } else {
          uint32_t loc = (sec - 1) / DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN);
          if (loc >= meta_.npoints) continue;
          uint32_t id = loc;
          if (!loc2id_.empty() && loc < loc2id_.size()) id = loc2id_[loc];
          if (id >= meta_.npoints) continue;
          DiskNode<T> node = node_from_page(sector_buf, 0);
          full_adj_->set_adj(id, node.nnbrs, node.nbrs);
        }
      }
      pipeann::aligned_free(sector_buf);
      LOG(INFO) << "FullAdjIndex built: " << meta_.npoints << " nodes, "
                << full_adj_->memory_gb() << " GB";

      // Save for future runs
      full_adj_->save(adj_path);

      if (lazy_adj) {
        // Re-open as lazy mmap (free the in-memory vector)
        full_adj_->load_mmap(adj_path, false);
      }
    }

    void load_spmat_filter_store(const std::string &path) {
      spmat_filter_ = std::make_unique<SpmatFilterStore>();
      spmat_filter_->load(path);
    }

    // Unified filter check: works with either single-label or multi-label store
    bool check_filter(uint32_t node_id, const void *filter_data) const {
      if (spmat_filter_ && spmat_filter_->loaded()) {
        return spmat_filter_->passes(node_id, filter_data);
      }
      if (filter_store_ && filter_store_->loaded() && filter_data) {
        uint8_t ql = *(const uint8_t *)filter_data;
        return filter_store_->passes(node_id, ql);
      }
      return true;  // no filter loaded
    }

    FilterStore  *get_filter_store() const { return filter_store_.get(); }
    SpmatFilterStore *get_spmat_filter() const { return spmat_filter_.get(); }
    FullAdjIndex *get_full_adj()     const { return full_adj_.get(); }

    // Tunneling access via FullAdjIndex
    bool tunneling_loaded() const {
      return full_adj_ && full_adj_->loaded();
    }
    void tunneling_get_adj(uint32_t id, uint32_t &nnbrs, const uint32_t *&nbrs) const {
      if (full_adj_ && full_adj_->loaded()) {
        full_adj_->get_adj(id, nnbrs, nbrs);
      } else {
        nnbrs = 0; nbrs = nullptr;
      }
    }
    void set_filter_precheck(bool v)       { filter_precheck_ = v; }
    void set_early_filter_check(bool v)    { early_filter_check_ = v; }
    void set_fdiskann_filter(bool v)       { fdiskann_filter_ = v; }

    // Filtered-DiskANN: compute per-label medoids following official DiskANN.
    // Each label gets a start-point node that carries that label, so filtered
    // beam_search begins in the relevant graph region instead of the global
    // entry point.  Official approach: random sample + load-balance across
    // labels (Algorithm 2, "FindMedoid" in WWW'23 paper).
    void compute_fdiskann_medoids() {
      if (!filter_store_ || !filter_store_->loaded()) return;
      fdiskann_medoids_.clear();

      // Group node IDs by label
      std::unordered_map<uint8_t, std::vector<uint32_t>> label_nodes;
      for (uint32_t i = 0; i < (uint32_t)filter_store_->size(); i++) {
        label_nodes[filter_store_->get_label(i)].push_back(i);
      }

      // Official DiskANN: sample up to 25 candidates per label, pick the one
      // with the lowest reuse count (load-balancing heuristic).  With one
      // medoid per label the reuse count is always 0, so we simply pick the
      // middle element of the sample (more likely to be graph-central than
      // the very first or last node).
      std::mt19937 rng(42);
      constexpr size_t kSampleSize = 25;
      for (auto &[label, nodes] : label_nodes) {
        if (nodes.empty()) continue;
        size_t n = std::min(kSampleSize, nodes.size());
        // Partial shuffle: move n random elements to front
        for (size_t i = 0; i < n; i++) {
          std::uniform_int_distribution<size_t> dist(i, nodes.size() - 1);
          std::swap(nodes[i], nodes[dist(rng)]);
        }
        // Pick the middle of the sample
        fdiskann_medoids_[label] = nodes[n / 2];
      }

      LOG(INFO) << "Filtered-DiskANN medoids: " << fdiskann_medoids_.size() << " labels";
      for (auto &[label, medoid] : fdiskann_medoids_) {
        LOG(INFO) << "  label=" << (int)label
                  << " medoid=" << medoid
                  << " (of " << label_nodes[label].size() << " nodes)";
      }
    }

    uint32_t get_fdiskann_medoid(uint8_t label) const {
      auto it = fdiskann_medoids_.find(label);
      return (it != fdiskann_medoids_.end()) ? it->second : meta_.entry_point_id;
    }

   private:
    // Background insert I/O commit.
    struct BgTask {
      QueryBuffer<T> *thread_data;
      std::vector<IORequest> writes;
      std::vector<uint64_t> pages_to_unlock;
      std::vector<uint64_t> pages_to_deref;
      bool terminate = false;
    };

    using PageArr = std::vector<uint32_t>;

    static constexpr int kBgIOThreads = 1;

    // Derived/runtime metadata not stored in SSDIndexMetadata.
    uint64_t aligned_dim = 0;
    uint64_t size_per_io = 0;

    // File reader and index file path.
    std::shared_ptr<AlignedFileReader> &reader;
    std::string disk_index_file;

    // Neighbor handler.
    AbstractNeighbor<T> *nbr_handler;

    // Distance comparator.
    pipeann::Metric metric;
    std::shared_ptr<Distance<T>> dist_cmp;

    // Update-related parameters.
    IndexBuildParameters params;

    // Thread-specific scratch buffers.
    ConcurrentQueue<QueryBuffer<T> *> thread_data_queue;
    std::vector<QueryBuffer<T> *> thread_data_bufs;  // pre-allocated thread data
    uint64_t max_nthreads;

    // Background I/O threads for insert.
    ConcurrentQueue<BgTask *> bg_tasks = ConcurrentQueue<BgTask *>(nullptr);
    std::thread *bg_io_thread_[kBgIOThreads]{nullptr};

    // Locking tables for concurrency control.
    pipeann::SparseLockTable<uint64_t> page_lock_table;
    pipeann::SparseLockTable<uint64_t> vec_lock_table;
    pipeann::SparseLockTable<uint64_t> page_idx_lock_table;
    pipeann::SparseLockTable<uint64_t> idx_lock_table;
    std::shared_mutex merge_lock;  // serve search during merge.

    // Page search mode flag.
    bool use_page_search_ = true;

    // ID to location mapping.
    // Concurrency control is done in lock_idx.
    // Only resize should be protected.
    std::vector<uint32_t> id2loc_;
    pipeann::ReaderOptSharedMutex id2loc_resize_mu_;

    // Location to ID mapping.
    // If nnodes_per_sector >= 1, page_layout[i * nnodes_per_sector + j] is the id of the j-th node in the i-th page.
    // ElseIf nnodes_per_sector == 0, page_layout[i] is the id of the i-th node (starting from loc_sector_no(i)).
    std::vector<uint32_t> loc2id_;
    pipeann::ReaderOptSharedMutex loc2id_resize_mu_;
    std::mutex alloc_lock;
    ConcurrentQueue<uint32_t> empty_pages = ConcurrentQueue<uint32_t>(kInvalidID);

    // Tag support.
    // If ID == tag, then it is not stored.
    libcuckoo::cuckoohash_map<uint32_t, TagT> tags;

    // Flags.
    bool load_flag = false;    // already loaded.
    bool enable_tags = false;  // support for tags and dynamic indexing

    // Hot node cache (nullptr when disabled; used by mode=4 cached pipe search).
    std::unique_ptr<HotNodeCache<T>> hot_cache_;

    // Filter-aware IO (mode=8): pre-IO filter check + tunneling
    std::unique_ptr<FilterStore>  filter_store_;         // single-label
    std::unique_ptr<SpmatFilterStore> spmat_filter_;     // multi-label (YFCC)
    std::unique_ptr<FullAdjIndex> full_adj_;
    bool filter_precheck_ = false;  // set true only for mode=8
    bool early_filter_check_ = false; // mode=9: check filter after IO, skip exact dist if non-matching
    bool fdiskann_filter_ = false; // Filtered-DiskANN: hard-filter non-matching neighbors from candidate set
    std::unordered_map<uint8_t, uint32_t> fdiskann_medoids_; // per-label start points

    void init_buffers(uint64_t nthreads);
    void destroy_buffers();

    // Load id2loc and loc2id (i.e., page_layout), to support index reordering.
    void load_page_layout(const std::string &index_prefix, const uint64_t nnodes_per_sector = 0,
                          const uint64_t num_points = 0);

    void load_tags(const std::string &tag_file, size_t offset = 0);

    // Direct insert related.
    void do_beam_search(const T *vec, uint32_t mem_L, uint32_t Lsize, const uint32_t beam_width,
                        std::vector<Neighbor> &expanded_nodes_info, tsl::robin_map<uint32_t, T *> *coord_map = nullptr,
                        T *coord_buf = nullptr, QueryStats *stats = nullptr,
                        tsl::robin_set<uint32_t> *exclude_nodes = nullptr, bool dyn_search_l = true,
                        std::vector<uint64_t> *passthrough_page_ref = nullptr,
                        const void *filter_data = nullptr);

    // Background I/O thread function.
    void bg_io_thread();

    int get_vector_by_id(const uint32_t &id, T *vector);

    // ID, loc, page mapping.
    TagT id2tag(uint32_t id);

    uint32_t id2loc(uint32_t id);
    void set_id2loc(uint32_t id, uint32_t loc);
    uint64_t id2page(uint32_t id);

    uint32_t loc2id(uint32_t loc);
    void set_loc2id(uint32_t loc, uint32_t id);
    void erase_loc2id(uint32_t loc);

    PageArr get_page_layout(uint32_t page_no);

    void erase_and_set_loc(const std::vector<uint64_t> &old_locs, const std::vector<uint64_t> &new_locs,
                           const std::vector<uint32_t> &new_ids);
    // Returns <loc, need_read>.
    std::vector<uint64_t> alloc_loc(int n, const std::vector<uint64_t> &hint_pages,
                                    std::set<uint64_t> &page_need_to_read);
    void verify_id2loc();

    // lock-related.
    void lock_vec(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                  const std::vector<uint32_t> &neighbors, bool rd = false);
    void unlock_vec(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                    const std::vector<uint32_t> &neighbors);

    std::vector<uint32_t> get_to_lock_idx(uint32_t target, const std::vector<uint32_t> &neighbors);

    // Lock the mapping for target/page if use_page_search == false/true.
    std::vector<uint32_t> lock_idx(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                                   const std::vector<uint32_t> &neighbors, bool rd = false);
    void unlock_idx(pipeann::SparseLockTable<uint64_t> &lock_table, const std::vector<uint32_t> &to_lock);
    void unlock_idx(pipeann::SparseLockTable<uint64_t> &lock_table, const uint32_t &to_lock);

    // Two-level lock, as id2page may change before and after grabbing the lock.
    std::vector<uint32_t> lock_page_idx(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                                        const std::vector<uint32_t> &neighbors, bool rd = false);
    void unlock_page_idx(pipeann::SparseLockTable<uint64_t> &lock_table, const std::vector<uint32_t> &to_lock);
  };
}  // namespace pipeann
