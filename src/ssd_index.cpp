#include "aligned_file_reader.h"
#include "ssd_index.h"
#include <malloc.h>

#include <omp.h>
#include <cmath>
#include "distance.h"
#include "nbr/nbr.h"
#include "ssd_index_defs.h"
#include "utils/timer.h"
#include "utils.h"

#include <filesystem>
#include <unistd.h>
#include <sys/syscall.h>
#include "utils/tsl/robin_set.h"

namespace pipeann {
  template<typename T, typename TagT>
  SSDIndex<T, TagT>::SSDIndex(pipeann::Metric m, std::shared_ptr<AlignedFileReader> &file_reader,
                              AbstractNeighbor<T> *nbr_handler, bool tags, IndexBuildParameters *parameters)
      : reader(file_reader), nbr_handler(nbr_handler), metric(m), enable_tags(tags) {
    this->dist_cmp.reset(pipeann::get_distance_function<T>(m));

    if (parameters != nullptr) {
      this->params = *parameters;
      params.print();
    }
    LOG(INFO) << "Use " << nbr_handler->get_name() << " as neighbor handler, metric: " << get_metric_str(m);
  }

  template<typename T, typename TagT>
  SSDIndex<T, TagT>::~SSDIndex() {
    LOG(INFO) << "Lock table size: " << this->idx_lock_table.size();
    LOG(INFO) << "Page cache size: " << pipeann::cache.cache.size();

    if (load_flag) {
      this->destroy_buffers();
      reader->close();
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::copy_index(const std::string &prefix_in, const std::string &prefix_out) {
    LOG(INFO) << "Copying disk index from " << prefix_in << " to " << prefix_out;
    std::filesystem::copy(prefix_in + "_disk.index", prefix_out + "_disk.index",
                          std::filesystem::copy_options::overwrite_existing);
    if (file_exists(prefix_in + "_disk.index.tags")) {
      std::filesystem::copy(prefix_in + "_disk.index.tags", prefix_out + "_disk.index.tags",
                            std::filesystem::copy_options::overwrite_existing);
    } else {
      // remove the original tags.
      std::filesystem::remove(prefix_out + "_disk.index.tags");
    }

    // nbr.
    this->nbr_handler->load(prefix_in.c_str());
    this->nbr_handler->save(prefix_out.c_str());

    // partition data
    if (file_exists(prefix_in + "_partition.bin.aligned")) {
      std::filesystem::copy(prefix_in + "_partition.bin.aligned", prefix_out + "_partition.bin.aligned",
                            std::filesystem::copy_options::overwrite_existing);
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::init_buffers(uint64_t n_threads) {
    uint64_t n_buffers = n_threads * 2;
    LOG(INFO) << "Init buffers for " << n_threads << " threads, setup " << n_buffers << " buffers.";
    this->thread_data_queue.null_T = nullptr;
    for (uint64_t i = 0; i < n_buffers; i++) {
      QueryBuffer<T> *data = new QueryBuffer<T>();
      this->init_query_buf(*data);
      this->thread_data_bufs.push_back(data);
      this->thread_data_queue.push(data);
      this->reader->register_buf(data->sector_scratch, MAX_N_SECTOR_READS * SECTOR_LEN, 0);
    }

#ifndef READ_ONLY_TESTS
    // background thread.
    LOG(INFO) << "Setup " << kBgIOThreads << " background I/O threads for insert...";
    for (int i = 0; i < kBgIOThreads; ++i) {
      bg_io_thread_[i] = new std::thread(&SSDIndex<T, TagT>::bg_io_thread, this);
      bg_io_thread_[i]->detach();
    }
#endif
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::destroy_buffers() {
#ifndef READ_ONLY_TESTS
    for (int i = 0; i < kBgIOThreads; ++i) {
      if (bg_io_thread_[i] != nullptr) {
        auto bg_task = new BgTask{
            .thread_data = nullptr, .writes = {}, .pages_to_unlock = {}, .pages_to_deref = {}, .terminate = true};
        bg_tasks.push(bg_task);
        bg_tasks.push_notify_all();
        bg_io_thread_[i] = nullptr;
      }
    }
#endif

    while (!this->thread_data_bufs.empty()) {
      auto buf = this->thread_data_bufs.back();
      pipeann::aligned_free((void *) buf->coord_scratch);
      pipeann::aligned_free((void *) buf->sector_scratch);
      pipeann::aligned_free((void *) buf->nbr_vec_scratch);
      pipeann::aligned_free((void *) buf->nbr_ctx_scratch);
      pipeann::aligned_free((void *) buf->aligned_dist_scratch);
      pipeann::aligned_free((void *) buf->aligned_query_T);
      this->thread_data_bufs.pop_back();
      this->thread_data_queue.pop();
      delete buf;
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_mem_index(const std::string &mem_index_path) {
    if (!this->load_flag) {
      LOG(ERROR) << "Index is not loaded. Please load the index first.";
      exit(-1);
    }
    mem_index_ = std::make_unique<pipeann::Index<T, uint32_t>>(this->metric, this->meta_.data_dim);
    mem_index_->load(mem_index_path.c_str());
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::load(const char *index_prefix, uint32_t num_threads, bool use_page_search) {
    std::string disk_index_file = std::string(index_prefix) + "_disk.index";
    this->disk_index_file = disk_index_file;

    SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(disk_index_file);
    this->init_metadata(meta);

    // load nbrs (e.g., PQ)
    nbr_handler->load(index_prefix);

    // read index metadata
    // open AlignedFileReader handle to index_file
    if (!file_exists(disk_index_file)) {
      LOG(ERROR) << "Index file " << disk_index_file << " does not exist!";
      exit(-1);
    }

    this->destroy_buffers();  // in case of re-init.
    reader->open(disk_index_file, true, false);
    this->init_buffers(num_threads);
    this->max_nthreads = num_threads;

    // load page layout.
    this->use_page_search_ = use_page_search;
    this->load_page_layout(index_prefix, meta_.nnodes_per_sector, meta_.npoints);

    // load tags
    if (this->enable_tags) {
      std::string tag_file = disk_index_file + ".tags";
      LOG(INFO) << "Loading tags from " << tag_file;
      this->load_tags(tag_file);
    }

    load_flag = true;
    LOG(INFO) << "SSDIndex loaded successfully.";
    return 0;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_page_layout(const std::string &index_prefix, const uint64_t nnodes_per_sector,
                                           const uint64_t num_points) {
    std::string partition_file = index_prefix + "_partition.bin.aligned";
    id2loc_.resize(num_points);  // pre-allocate space first.
    loc2id_.resize(cur_loc);     // pre-allocate space first.

    if (file_exists(partition_file)) {
      LOG(INFO) << "Loading partition file " << partition_file;
      std::ifstream part(partition_file);
      uint64_t C, partition_nums, nd;
      part.read((char *) &C, sizeof(uint64_t));
      part.read((char *) &partition_nums, sizeof(uint64_t));
      part.read((char *) &nd, sizeof(uint64_t));
      if (nnodes_per_sector <= 1 || C != nnodes_per_sector) {
        // graph reordering is useful only when nnodes_per_sector > 1.
        LOG(ERROR) << "partition information not correct.";
        exit(-1);
      }
      LOG(INFO) << "Partition meta: C: " << C << " partition_nums: " << partition_nums;

      uint64_t page_offset = loc_sector_no(0);
      auto st = std::chrono::high_resolution_clock::now();

      constexpr uint64_t n_parts_per_read = 1024 * 1024;
      std::vector<unsigned> part_buf(n_parts_per_read * (1 + nnodes_per_sector));
      for (uint64_t p = 0; p < partition_nums; p += n_parts_per_read) {
        uint64_t nxt_p = std::min(p + n_parts_per_read, partition_nums);
        part.read((char *) part_buf.data(), sizeof(unsigned) * n_parts_per_read * (1 + nnodes_per_sector));
#pragma omp parallel for schedule(dynamic)
        for (uint64_t i = p; i < nxt_p; ++i) {
          uint32_t base = (i - p) * (1 + nnodes_per_sector);
          uint32_t s = part_buf[base];  // size of this partition
          for (uint32_t j = 0; j < s; ++j) {
            uint64_t id = part_buf[base + 1 + j];
            uint64_t loc = i * nnodes_per_sector + j;
            id2loc_[id] = loc;
            loc2id_[loc] = id;
          }
          for (uint32_t j = s; j < nnodes_per_sector; ++j) {
            loc2id_[i * nnodes_per_sector + j] = kInvalidID;
          }
        }
      }
      auto et = std::chrono::high_resolution_clock::now();
      LOG(INFO) << "Page layout loaded in " << std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count()
                << " ms";
    } else {
      LOG(INFO) << partition_file << " does not exist, use equal partition mapping";
// use equal mapping for id2loc and page_layout.
#ifndef NO_MAPPING
#pragma omp parallel for
      for (size_t i = 0; i < meta_.npoints; ++i) {
        id2loc_[i] = i;
        loc2id_[i] = i;
      }
      for (size_t i = meta_.npoints; i < this->cur_loc; ++i) {
        loc2id_[i] = kInvalidID;
      }
#endif
    }
    LOG(INFO) << "Page layout loaded.";
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_tags(const std::string &tag_file_name, size_t offset) {
    size_t tag_num, tag_dim;
    std::vector<TagT> tag_v;
    this->tags.clear();

    if (!file_exists(tag_file_name)) {
      LOG(INFO) << "Tags file not found. Using equal mapping";
      // Equal mapping are by default eliminated in tags map.
    } else {
      LOG(INFO) << "Load tags from existing file: " << tag_file_name;
      pipeann::load_bin<TagT>(tag_file_name, tag_v, tag_num, tag_dim, offset);
      tags.reserve(tag_v.size());

#pragma omp parallel for num_threads(max_nthreads)
      for (size_t i = 0; i < tag_num; ++i) {
        tags.insert_or_assign(i, tag_v[i]);
      }
      LOG(INFO) << "Loaded " << tags.size() << " tags";
    }
  }

  template<typename T, typename TagT>
  Index<T, TagT> *SSDIndex<T, TagT>::load_to_mem(const std::string &filename) {
    std::string disk_index_file = std::string(filename) + "_disk.index";
    SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(disk_index_file);
    this->init_metadata(meta);  // ensure that loc_sector_no is correct.

    Index<T, TagT> *mem_index = new Index<T, TagT>(this->metric, meta.data_dim);

    mem_index->_ep = meta.entry_point;
    mem_index->range = meta.range;
    mem_index->_nd = meta.npoints;
    mem_index->resize(meta.npoints);

    reader->open(disk_index_file, true, false);

    const uint64_t kLocsPerRead = meta_.nnodes_per_sector > 0 ? ROUND_UP(131072, meta_.nnodes_per_sector)
                                                              : 131072;  // process this many locs per iteration

    // Calculate buffer size based on whether we have large or small nodes
    uint64_t sectors_per_batch = meta_.nnodes_per_sector > 0
                                     ? kLocsPerRead / meta_.nnodes_per_sector
                                     : kLocsPerRead * DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN);
    uint64_t buf_size_per_batch = sectors_per_batch * SECTOR_LEN;

    char *buf;
    pipeann::alloc_aligned((void **) &buf, buf_size_per_batch, SECTOR_LEN);

    for (uint64_t loc_st = 0; loc_st < meta.npoints; loc_st += kLocsPerRead) {
      uint64_t loc_ed = std::min(meta.npoints, loc_st + kLocsPerRead);

      // Calculate sector range to read for [loc_st, loc_ed)
      uint64_t st_sector = loc_sector_no(loc_st);
      uint64_t ed_sector = loc_sector_no(loc_ed > 0 ? loc_ed - 1 : 0);
      uint64_t n_sectors_to_read = ed_sector - st_sector + 1;

      std::vector<IORequest> read_reqs;
      read_reqs.push_back(IORequest(st_sector * SECTOR_LEN, n_sectors_to_read * SECTOR_LEN, buf, 0, 0));
      reader->read(read_reqs, reader->get_ctx(), false);

#pragma omp parallel for
      for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
        uint64_t id = loc;
#pragma omp critical
        {
          mem_index->_location_to_tag[id] = id;
          mem_index->_tag_to_location[id] = id;
        }

        uint64_t loc_sector = loc_sector_no(loc);
        auto page_rbuf = buf + (loc_sector - st_sector) * SECTOR_LEN;
        DiskNode<T> node = node_from_page(page_rbuf, loc);

        // load data and nhood.
        memcpy(mem_index->_data.data() + id * meta.data_dim, node.coords, meta.data_dim * sizeof(T));
        std::vector<uint32_t> nhood;

        for (uint32_t i = 0; i < node.nnbrs; ++i) {
          nhood.push_back(node.nbrs[i]);
        }
        mem_index->_final_graph[id] = nhood;
      }

      LOG(INFO) << "Loaded " << loc_ed << "/" << meta.npoints << " nodes.";
    }
    return mem_index;
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::get_vector_by_id(const uint32_t &id, T *vector_coords) {
    if (!enable_tags) {
      LOG(INFO) << "Tags are disabled, cannot retrieve vector";
      return -1;
    }
    uint32_t pos = id;
    auto loc = id2loc(pos);

    size_t num_sectors = loc_sector_no(loc);
    std::ifstream disk_reader(disk_index_file.c_str(), std::ios::binary);
    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(size_per_io);
    disk_reader.seekg(SECTOR_LEN * num_sectors, std::ios::beg);
    disk_reader.read(sector_buf.get(), size_per_io);
    DiskNode<T> node = node_from_page(sector_buf.get(), loc);
    memcpy((void *) vector_coords, (void *) node.coords, meta_.data_dim * sizeof(T));
    return 0;
  }

  template<typename T, typename TagT>
  TagT SSDIndex<T, TagT>::id2tag(uint32_t id) {
#ifdef NO_MAPPING
    return id;  // use ID to replace tags.
#else
    TagT ret;
    if (tags.find(id, ret)) {
      return ret;
    } else {
      return id;
    }
#endif
  }

  template<typename T, typename TagT>
  uint32_t SSDIndex<T, TagT>::id2loc(uint32_t id) {
#ifdef NO_MAPPING
    return id;
#else
    id2loc_resize_mu_.lock_shared();
    if (unlikely(id >= id2loc_.size())) {
      LOG(ERROR) << "id " << id << " is out of range " << id2loc_.size();
      crash();
      return kInvalidID;
    }
    uint32_t ret = id2loc_[id];
    id2loc_resize_mu_.unlock_shared();
    return ret;
#endif
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::set_id2loc(uint32_t id, uint32_t loc) {
#ifdef NO_MAPPING
    return;
#else
    if (unlikely(id >= id2loc_.size())) {
      id2loc_resize_mu_.lock();
      if (likely(id >= id2loc_.size())) {
        id2loc_.resize(1.5 * id);
        LOG(INFO) << "Resize id2loc_ to " << id2loc_.size();
      }
      id2loc_resize_mu_.unlock();
    }
    // Here, we do not grab any locks. But no matter:
    // Here, the id2loc_.size() must > id (as it only increases).
    // So, we only need to ensure that no concurrent resize happens (use-after-free).
    id2loc_resize_mu_.lock_shared();
    id2loc_[id] = loc;
    id2loc_resize_mu_.unlock_shared();
#endif
  }

  template<typename T, typename TagT>
  uint64_t SSDIndex<T, TagT>::id2page(uint32_t id) {
    uint32_t loc = id2loc(id);
    if (loc == kInvalidID) {
      return kInvalidID;
    }
    return loc_sector_no(loc);
  }

  template<typename T, typename TagT>
  uint32_t SSDIndex<T, TagT>::loc2id(uint32_t loc) {
    loc2id_resize_mu_.lock_shared();
    if (unlikely(loc > loc2id_.size())) {
      LOG(ERROR) << "loc " << loc << " is out of range " << loc2id_.size();
      crash();
      return kInvalidID;
    }
    uint32_t ret = loc2id_[loc];
    loc2id_resize_mu_.unlock_shared();
    return ret;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::set_loc2id(uint32_t loc, uint32_t id) {
    if (unlikely(loc >= loc2id_.size())) {
      loc2id_resize_mu_.lock();
      if (likely(loc >= loc2id_.size())) {
        loc2id_.resize(1.5 * loc);
        LOG(INFO) << "Resize loc2id_ to " << loc2id_.size();
      }
      loc2id_resize_mu_.unlock();
    }
    loc2id_resize_mu_.lock_shared();
    loc2id_[loc] = id;
    loc2id_resize_mu_.unlock_shared();
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::erase_loc2id(uint32_t loc) {
    loc2id_resize_mu_.lock_shared();
    loc2id_[loc] = kInvalidID;
    uint32_t st = sector_to_loc(loc_sector_no(loc), 0);
    bool empty = true;
    for (uint32_t i = st; i < st + meta_.nnodes_per_sector; ++i) {
      if (loc2id_[i] != kInvalidID) {
        empty = false;
        break;
      }
    }
    if (empty) {
      empty_pages.push(loc_sector_no(loc));
    }
    uint32_t page = loc_sector_no(loc);
    uint32_t offset = loc % meta_.nnodes_per_sector;
    loc2id_resize_mu_.unlock_shared();
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::lock_vec(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                                   const std::vector<uint32_t> &neighbors, bool rd) {
    std::vector<uint32_t> to_lock;
    to_lock.assign(neighbors.begin(), neighbors.end());
    if (target != kInvalidID)
      to_lock.push_back(target);
    std::sort(to_lock.begin(), to_lock.end());
    for (auto &id : to_lock) {
      rd ? lock_table.rdlock(id) : lock_table.wrlock(id);
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::unlock_vec(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                                     const std::vector<uint32_t> &neighbors) {
    if (target != kInvalidID)
      lock_table.unlock(target);
    for (auto &id : neighbors) {
      lock_table.unlock(id);
    }
  }

  template<typename T, typename TagT>
  std::vector<uint32_t> SSDIndex<T, TagT>::get_to_lock_idx(uint32_t target, const std::vector<uint32_t> &neighbors) {
    std::vector<uint32_t> to_lock;
    to_lock.assign(neighbors.begin(), neighbors.end());
    if (target != kInvalidID) {
      to_lock.push_back(target);
    }
    // Sort and deduplicate.
    std::sort(to_lock.begin(), to_lock.end());
    auto last = std::unique(to_lock.begin(), to_lock.end());
    to_lock.erase(last, to_lock.end());
    return to_lock;
  }

  // Lock the mapping for target/page if use_page_search == false/true.
  template<typename T, typename TagT>
  std::vector<uint32_t> SSDIndex<T, TagT>::lock_idx(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                                                    const std::vector<uint32_t> &neighbors, bool rd) {
#ifndef READ_ONLY_TESTS
    std::vector<uint32_t> to_lock = get_to_lock_idx(target, neighbors);
    for (auto &id : to_lock) {
      rd ? lock_table.rdlock(id) : lock_table.wrlock(id);
    }
    return to_lock;
#else
    return std::vector<uint32_t>();
#endif
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::unlock_idx(pipeann::SparseLockTable<uint64_t> &lock_table,
                                     const std::vector<uint32_t> &to_lock) {
#ifndef READ_ONLY_TESTS
    for (auto &id : to_lock) {
      lock_table.unlock(id);
    }
#endif
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::unlock_idx(pipeann::SparseLockTable<uint64_t> &lock_table, const uint32_t &to_lock) {
#ifndef READ_ONLY_TESTS
    lock_table.unlock(to_lock);
#endif
  }

  // Two-level lock, as id2page may change before and after grabbing the lock.
  template<typename T, typename TagT>
  std::vector<uint32_t> SSDIndex<T, TagT>::lock_page_idx(pipeann::SparseLockTable<uint64_t> &lock_table,
                                                         uint32_t target, const std::vector<uint32_t> &neighbors,
                                                         bool rd) {
#ifndef READ_ONLY_TESTS
    if (!use_page_search_) {
      return std::vector<uint32_t>();
    }
    std::vector<uint32_t> to_lock(neighbors.begin(), neighbors.end());
    if (target != kInvalidID) {
      to_lock.push_back(target);
    }

    for (size_t i = 0; i < to_lock.size(); ++i) {
      to_lock[i] = id2page(to_lock[i]);
    }

    // Sort and deduplicate.
    std::sort(to_lock.begin(), to_lock.end());
    auto last = std::unique(to_lock.begin(), to_lock.end());
    to_lock.erase(last, to_lock.end());

    for (auto &id : to_lock) {
      rd ? lock_table.rdlock(id) : lock_table.wrlock(id);
    }
    return to_lock;
#else
    return std::vector<uint32_t>();
#endif
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::unlock_page_idx(pipeann::SparseLockTable<uint64_t> &lock_table,
                                          const std::vector<uint32_t> &to_lock) {
#ifndef READ_ONLY_TESTS
    if (!use_page_search_) {
      return;
    }
    for (auto &id : to_lock) {
      lock_table.unlock(id);
    }
#endif
  }

  template<typename T, typename TagT>
  typename SSDIndex<T, TagT>::PageArr SSDIndex<T, TagT>::get_page_layout(uint32_t page_no) {
    loc2id_resize_mu_.lock_shared();
    PageArr ret;
    auto st = sector_to_loc(page_no, 0);
    auto ed = meta_.nnodes_per_sector == 0 ? st + 1 : st + meta_.nnodes_per_sector;
    for (uint32_t i = st; i < ed; ++i) {
      ret.push_back(loc2id_[i]);
    }
    loc2id_resize_mu_.unlock_shared();
    return ret;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::erase_and_set_loc(const std::vector<uint64_t> &old_locs,
                                            const std::vector<uint64_t> &new_locs,
                                            const std::vector<uint32_t> &new_ids) {
    std::lock_guard<std::mutex> lock(alloc_lock);
    for (uint32_t i = 0; i < new_locs.size(); ++i) {
      set_loc2id(new_locs[i], new_ids[i]);
    }
    for (auto &l : old_locs) {
      erase_loc2id(l);
    }
  }

  // Returns <loc, need_read>.
  template<typename T, typename TagT>
  std::vector<uint64_t> SSDIndex<T, TagT>::alloc_loc(int n, const std::vector<uint64_t> &hint_pages,
                                                     std::set<uint64_t> &page_need_to_read) {
    std::lock_guard<std::mutex> lock(alloc_lock);
    std::vector<uint64_t> ret;
    int cur = 0;
    // Reuse.
    uint32_t threshold = (meta_.nnodes_per_sector + INDEX_SIZE_FACTOR - 1) / INDEX_SIZE_FACTOR;

    // 1. Use empty pages.
    uint32_t empty_page = kInvalidID;
    while ((empty_page = empty_pages.pop()) != kInvalidID) {
#ifdef NO_POLLUTE_ORIGINAL
      if (empty_page < loc_sector_no(meta_.npoints)) {
        continue;
      }
#endif
      auto st = sector_to_loc(empty_page, 0);
      auto ed = meta_.nnodes_per_sector == 0 ? st + 1 : st + meta_.nnodes_per_sector;
      for (uint32_t i = st; i < ed; ++i) {
        if (unlikely(loc2id_[i] != kInvalidID)) {
          LOG(ERROR) << "Page " << empty_page << " is not empty " << i << " " << loc2id_[i];
          crash();
        }
        loc2id_[i] = kAllocatedID;
        ret.push_back(i);
        ++cur;
        if (cur == n) {
          return ret;
        }
      }
    }

    // 2. Use hint pages.
    for (auto &p : hint_pages) {
#ifdef NO_POLLUTE_ORIGINAL
      if (p < loc_sector_no(meta_.npoints)) {
        continue;
      }
#endif
      // First, see the number of holes.
      uint32_t cnt = 0;
      auto st = sector_to_loc(p, 0);
      auto ed = meta_.nnodes_per_sector == 0 ? st + 1 : st + meta_.nnodes_per_sector;
      for (uint32_t i = st; i < ed; ++i) {
        if (loc2id_[i] == kInvalidID) {
          cnt++;
        }
      }
      if (cnt < threshold) {
        continue;
      }
      // Second, allocate them.
      if (cnt < meta_.nnodes_per_sector) {
        page_need_to_read.insert(p);
      }
      for (uint32_t i = st; i < ed; ++i) {
        if (loc2id_[i] == kInvalidID) {
          loc2id_[i] = kAllocatedID;
          ret.push_back(i);
          ++cur;
          if (cur == n) {
            return ret;
          }
        }
      }
    }

    // 3. Use new pages.
    int remaining = n - cur;
    for (int i = 0; i < remaining; i++) {
      set_loc2id(cur_loc + i, kAllocatedID);  // auto resize.
      ret.push_back(cur_loc + i);
    }

    // Ensure that cur_loc is aligned.
    // The hole will eventually be recycled using either empty page queue or hint pages.
    cur_loc += remaining;
    while (meta_.nnodes_per_sector != 0 && cur_loc % meta_.nnodes_per_sector != 0) {
      set_loc2id(cur_loc++, kInvalidID);  // auto resize.
    }
    return ret;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::verify_id2loc() {
    // Verify id -> loc -> id map.
    LOG(INFO) << "ID2loc size: " << id2loc_.size() << ", cur_loc: " << cur_loc.load() << ", cur_id: " << cur_id
              << ", nnodes_per_sector: " << meta_.nnodes_per_sector;
    for (uint32_t i = 0; i < cur_id; ++i) {
      auto loc = id2loc(i);
      if (unlikely(loc >= cur_loc.load())) {
        LOG(ERROR) << "ID2loc inconsistency at ID: " << i << ", loc: " << loc << ", cur_loc: " << cur_loc.load();
        crash();
      }
      if (unlikely(loc2id(loc) != i)) {
        LOG(ERROR) << "ID2loc inconsistency at ID: " << i << ", loc: " << id2loc(i)
                   << ", loc2id: " << loc2id(id2loc(i));
        crash();
      }
    }

    LOG(INFO) << "ID2loc consistency check passed.";

    // Verify loc2id do not contain duplicate ids.
    for (uint32_t i = 0; i < cur_loc; ++i) {
      auto id = loc2id(i);
      if (id != kInvalidID && id != kAllocatedID) {
        uint32_t loc = id2loc(id);
        if (unlikely(loc != i)) {
          LOG(ERROR) << "loc2id inconsistency at loc: " << i << ", id: " << id << ", loc: " << loc;
        }
      }
    }
    LOG(INFO) << "loc2ID consistency check passed.";
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
