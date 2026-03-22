#include "utils/timer.h"
#include "utils/tsl/robin_set.h"
#include "utils.h"
#include "dynamic_index.h"
#include <csignal>
#include <cstdint>
#include <mutex>
#include <vector>
#include <utility>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <omp.h>
#include <shared_mutex>
#include <string>

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <gperftools/malloc_extension.h>

#include "ssd_index.h"

#include "linux_aligned_file_reader.h"

namespace pipeann {
  template<typename T, typename TagT>
  DynamicSSDIndex<T, TagT>::DynamicSSDIndex(IndexBuildParameters &parameters, const std::string disk_prefix_in,
                                            const std::string disk_prefix_out, Distance<T> *dist,
                                            pipeann::Metric dist_metric, int search_mode, bool use_mem_index) {
    // check if file exists.
    if (!file_exists(disk_prefix_in + "_disk.index")) {
      LOG(ERROR) << "Disk index file does not exist: " << disk_prefix_in << "_disk.index";
      exit(-1);
    }
    if (use_mem_index && !file_exists(disk_prefix_in + "_mem.index")) {
      LOG(ERROR) << "In-memory index file does not exist: " << disk_prefix_in << "_mem.index";
      exit(-1);
    }

    this->_dist_metric = dist_metric;
    this->journal = new pipeann::Journal<TagT>(disk_prefix_out + "_journal");

    _paras_disk = parameters;
    _num_threads = parameters.num_threads;
    _beamwidth = parameters.beam_width;

    _disk_index_prefix_in = disk_prefix_in;
    _disk_index_prefix_out = disk_prefix_out;
    _dist_comp = dist;

    reader.reset(new LinuxAlignedFileReader());
    AbstractNeighbor<T> *nbr_handler = new PQNeighbor<T>(this->_dist_metric);
    _disk_index = new pipeann::SSDIndex<T, TagT>(this->_dist_metric, reader, nbr_handler, true, &_paras_disk);

#ifndef NO_POLLUTE_ORIGINAL
    std::string disk_index_prefix_shadow = _disk_index_prefix_in + "_shadow";
    _disk_index->copy_index(_disk_index_prefix_in, disk_index_prefix_shadow);
    LOG(INFO) << "Copy disk index file to " << disk_index_prefix_shadow << "_disk.index";
    _disk_index_prefix_in = disk_index_prefix_shadow;
#endif

    if (search_mode == BEAM_SEARCH || search_mode == PAGE_SEARCH || search_mode == PIPE_SEARCH) {
      this->search_mode = search_mode;
    } else {
      LOG(ERROR) << "Invalid search mode: " << search_mode
                 << ". Must be one of BEAM_SEARCH, PAGE_SEARCH, or PIPE_SEARCH.";
      exit(-1);
    }
    bool use_page_search = (search_mode == PAGE_SEARCH);
    int res = _disk_index->load(_disk_index_prefix_in.c_str(), _num_threads, use_page_search);
    if (res != 0) {
      LOG(INFO) << "Failed to load disk index in DynamicSSDIndex constructor";
      exit(-1);
    }

    this->_use_mem_index = use_mem_index;
    if (use_mem_index) {
      std::string mem_index_path = disk_prefix_in + "_mem.index";  // use the original one.
      LOG(INFO) << "Use static in-memory index for acceleration, path: " << mem_index_path;
      _disk_index->load_mem_index(mem_index_path);
    }
  }

  template<typename T, typename TagT>
  DynamicSSDIndex<T, TagT>::~DynamicSSDIndex() {
    // put in destructor code
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::checkpoint() {
    // TODO(gh): checkpoint the index.
    journal->checkpoint();
  }

  template<typename T, typename TagT>
  int DynamicSSDIndex<T, TagT>::insert(const T *point, const TagT &tag) {
    std::shared_lock<std::shared_timed_mutex> lock(_merge_lock);  // prevent merge during insert
    journal->append(pipeann::TxType::kInsert, tag);
    auto *deletion_set = &deletion_sets[active_delete_set];
    return _disk_index->insert_in_place(point, tag, deletion_set);
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::search(const T *query, const uint64_t K, const uint32_t mem_L, const uint64_t search_L,
                                        const uint32_t beam_width, TagT *tags, float *distances, QueryStats *stats,
                                        bool dyn_search_l) {
    std::vector<TagT> result_tags(4096);
    std::vector<float> result_distances(4096);
    auto *deletion_set = &deletion_sets[active_delete_set];
    size_t n = 0;
    if (search_mode == BEAM_SEARCH) {
      n = _disk_index->beam_search(query, search_L, mem_L, search_L, result_tags.data(), result_distances.data(),
                                   beam_width, stats, deletion_set, dyn_search_l);
    } else if (search_mode == PAGE_SEARCH) {
      n = _disk_index->page_search(query, search_L, mem_L, search_L, result_tags.data(), result_distances.data(),
                                   beam_width, stats);
    } else if (search_mode == PIPE_SEARCH) {
      n = _disk_index->pipe_search(query, search_L, mem_L, search_L, result_tags.data(), result_distances.data(),
                                   beam_width, stats);
    }

    std::vector<std::pair<TagT, float>> best_vec;  // <tag, distance>
    for (size_t i = 0; i < n; i++) {
      best_vec.emplace_back(result_tags[i], result_distances[i]);
    }
    std::shared_lock<std::shared_timed_mutex> lock(delete_lock);
    size_t pos = 0;

    for (auto iter : best_vec) {
      if (deletion_sets[0].find(iter.first) == deletion_sets[0].end() &&
          deletion_sets[1].find(iter.first) == deletion_sets[1].end()) {
        tags[pos] = iter.first;
        distances[pos] = iter.second;
        pos++;
      }
      if (pos == K) {
        return;
      }
    }
    // LOG(INFO) << "Failed to find enough tags after " << i + 1 << " attempts";
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::lazy_delete(const TagT &tag) {
    std::unique_lock<std::shared_timed_mutex> lock(delete_lock);
    journal->append(pipeann::TxType::kDelete, tag);

    if (deletion_sets[active_delete_set].find(tag) == deletion_sets[active_delete_set].end()) {
      deletion_sets[active_delete_set].insert(tag);
      deleted_tags[active_delete_set].push_back(tag);
    }
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::final_merge(const uint32_t &nthreads, const uint32_t &n_sampled_nbrs) {
    std::unique_lock<std::shared_timed_mutex> lock(_merge_lock);  // only one merge at a time
    // _disk_index_in -> _disk_index_out
    // Before merge, only the active deletion set contains deletes.
    {
      std::unique_lock<std::shared_timed_mutex> lock(delete_lock);
      active_delete_set = !active_delete_set;
    }
    // During merge, both deletion_sets contain deletes.
    pipeann::Timer timer;
    merge(nthreads, n_sampled_nbrs);

    // After merge, clear the inactive deletion set (as they are already merged).
    // Only concurrent search & delete; no concurrent inserts as _merge_lock is held.
    {
      std::unique_lock<std::shared_timed_mutex> lock(delete_lock);
      deletion_sets[!active_delete_set].clear();
      deleted_tags[!active_delete_set].clear();
    }

    // TODO(gh): do we really need to reload disk index?
    std::swap(_disk_index_prefix_in, _disk_index_prefix_out);
    // reload the disk index
    _disk_index->reload(_disk_index_prefix_in.c_str(), _num_threads);
    LOG(INFO) << "Merge time : " << timer.elapsed() / 1000 << " ms";
    MallocExtension::instance()->ReleaseFreeMemory();  // Return free list to OS.
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::merge(const uint32_t &nthreads, const uint32_t &n_sampled_nbrs) {
    _disk_index->merge_deletes(_disk_index_prefix_in, _disk_index_prefix_out, deleted_tags[1 - active_delete_set],
                               deletion_sets[1 - active_delete_set], nthreads, n_sampled_nbrs);
  }

  template class DynamicSSDIndex<float>;
  template class DynamicSSDIndex<uint8_t>;
  template class DynamicSSDIndex<int8_t>;
}  // namespace pipeann
