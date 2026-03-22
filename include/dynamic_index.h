#pragma once

#include "utils/journal.h"
#include "utils/tsl/robin_set.h"
#include "ssd_index.h"
#include "index.h"
#include <atomic>
#include <limits>
#include <vector>
#include <cassert>
#include <shared_mutex>
#include <string>
#include <unordered_map>

namespace pipeann {

  template<typename T, typename TagT = uint32_t>
  class DynamicSSDIndex {
   public:
    /*
    Params:
    - parameters: IndexBuildParameters object with configuration of on-disk index.
    */
    DynamicSSDIndex(IndexBuildParameters &parameters, const std::string disk_prefix_in,
                    const std::string disk_prefix_out, Distance<T> *dist, pipeann::Metric disk_metric,
                    int search_mode = BEAM_SEARCH, bool use_mem_index = false);

    ~DynamicSSDIndex();

    void checkpoint();
    pipeann::Journal<TagT> *journal;

    // in-place
    int insert(const T *point, const TagT &tag);

    void search(const T *query, const uint64_t K, const uint32_t mem_L, const uint64_t search_L,
                const uint32_t beam_width, TagT *tags, float *distances, QueryStats *stats, bool dyn_search_l = true);

    void lazy_delete(const TagT &tag);

    void final_merge(const uint32_t &nthreads = 0,
                     const uint32_t &n_sampled_nbrs = std::numeric_limits<uint32_t>::max());

   private:
    void merge(const uint32_t &nthreads, const uint32_t &n_sampled_nbrs);

   public:
    size_t _dim;
    uint32_t _num_threads;  // search + insert + delete
    uint64_t _beamwidth;

    std::shared_ptr<AlignedFileReader> reader = nullptr;
    SSDIndex<T, TagT> *_disk_index = nullptr;

    pipeann::Metric _dist_metric;
    Distance<T> *_dist_comp;

    pipeann::IndexBuildParameters _paras_mem;
    pipeann::IndexBuildParameters _paras_disk;

    int active_delete_set = 0;            // reflects active _deletion_set
    std::shared_timed_mutex delete_lock;  // lock to access _deletion_set
    tsl::robin_set<TagT> deletion_sets[2];
    std::vector<TagT> deleted_tags[2];

    std::shared_timed_mutex _merge_lock;

    std::string _disk_index_prefix_in;
    std::string _disk_index_prefix_out;

    bool _use_mem_index = false;
    int search_mode = BEAM_SEARCH;
  };
}  // namespace pipeann
