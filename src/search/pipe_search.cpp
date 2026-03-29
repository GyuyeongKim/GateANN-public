#include "aligned_file_reader.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>
#ifndef USE_AIO
#include "liburing.h"
#endif

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include "utils/timer.h"
#include "utils/tsl/robin_set.h"
#include "utils.h"
#include "utils/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>

namespace pipeann {
  struct io_t {
    Neighbor nbr;
    unsigned page_id;
    unsigned loc;
    IORequest *read_req;
    bool operator>(const io_t &rhs) const {
      return nbr.distance > rhs.nbr.distance;
    }

    bool operator<(const io_t &rhs) const {
      return nbr.distance < rhs.nbr.distance;
    }

    bool finished() {
      return read_req->finished;
    }
  };

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::pipe_search(const T *query1, const uint64_t k_search, const uint32_t mem_L,
                                        const uint64_t l_search, TagT *res_tags, float *distances,
                                        const uint64_t beam_width, QueryStats *stats, AbstractSelector *selector,
                                        const void *filter_data, const uint64_t relaxed_monotonicity_l) {
    QueryBuffer<T> *query_buf = pop_query_buf(query1);
#ifdef USE_AIO
    void *ctx = reader->get_ctx();
#else
    void *ctx = reader->get_ctx(IORING_SETUP_SQPOLL);  // use SQ polling only for pipe search.
#endif

    if (beam_width > MAX_N_SECTOR_READS) {
      LOG(ERROR) << "Beamwidth can not be higher than MAX_N_SECTOR_READS";
      crash();
    }

    // relaxed_monotonicity mode: use relaxed monotonicity for termination besides Lmax.
    // - converge_size records #results when it first converges.
    // - When the number of results reaches converge_size + relaxed_monotonicity_l, the search terminates.
    // - L serves as the upper bound of candidate pool size now.
    int64_t converge_size = -1;  // -1 means not yet converged

    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    const T *query = query_buf->aligned_query_T;

    // reset query
    query_buf->reset();

    // pointers to current vector for comparison
    T *data_buf = query_buf->coord_scratch;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_buf->sector_scratch;
    float *dist_scratch = query_buf->aligned_dist_scratch;

    Timer query_timer;

    const unsigned effective_lsearch = l_search;
    std::vector<Neighbor> retset(mem_L + this->params.R + effective_lsearch * 10);
    auto &visited = *(query_buf->visited);
    unsigned cur_list_size = 0;

    std::vector<Neighbor> full_retset;
    full_retset.reserve(l_search * 10);


#ifndef OVERLAP_INIT
    nbr_handler->initialize_query(query, query_buf);
#endif

    auto compute_exact_dists_and_push = [&](const DiskNode<T> &node, const unsigned id) -> float {
      T *node_fp_coords_copy = data_buf;
      memcpy(node_fp_coords_copy, node.coords, meta_.data_dim * sizeof(T));
      float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);
      // Only add to full_retset if no filter or filter condition is satisfied.
      if (selector == nullptr || selector->is_member(id, filter_data, node.labels)) {
        full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
      }
      return cur_expanded_dist;
    };

    uint64_t n_computes = 0;
    auto compute_and_push_nbrs = [&](DiskNode<T> &node, unsigned &nk) {
      unsigned nbors_cand_size = 0;
      for (unsigned m = 0; m < node.nnbrs; ++m) {
        if (visited.find(node.nbrs[m]) == visited.end()) {
          node.nbrs[nbors_cand_size++] = node.nbrs[m];
          visited.insert(node.nbrs[m]);
        }
      }

      n_computes += nbors_cand_size;
      if (nbors_cand_size) {
        // auto cpu1_st = std::chrono::high_resolution_clock::now();
        nbr_handler->compute_dists(query_buf, node.nbrs, nbors_cand_size);
        for (unsigned m = 0; m < nbors_cand_size; ++m) {
          const int nbor_id = node.nbrs[m];
          const float nbor_dist = dist_scratch[m];
          // Filtered-DiskANN: hard-skip non-matching neighbors (not added to candidate set)
          if (this->fdiskann_filter_ && this->filter_store_ &&
              this->filter_store_->loaded() && filter_data != nullptr) {
            uint8_t q_label = *(const uint8_t *)filter_data;
            if (!this->filter_store_->passes(nbor_id, q_label)) {
              continue;
            }
          }
          if (stats != nullptr) {
            stats->n_cmps++;
          }

          if (nbor_dist >= retset[cur_list_size - 1].distance && (cur_list_size == effective_lsearch))
            continue;
          Neighbor nn(nbor_id, nbor_dist, true);
          // Return position in sorted list where nn inserted
          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);  // may be overflow in retset...
          if (cur_list_size < effective_lsearch) {
            ++cur_list_size;
            if (unlikely(cur_list_size >= retset.size())) {
              retset.resize(2 * cur_list_size);
            }
          }
          // nk logs the best position in the retset that was updated due to
          // neighbors of n.
          if (r < nk)
            nk = r;
        }
        // auto cpu1_ed = std::chrono::high_resolution_clock::now();
        // stats->cpu_us1 += std::chrono::duration_cast<std::chrono::microseconds>(cpu1_ed - cpu1_st).count();
      }
    };

    // Eager-specific neighbor expansion:
    // - Hot neighbors: exact distance (needed for chain quality; PQ underestimates corrupt DFS direction)
    // - Cold neighbors: PQ distance; they compete for slots in retset on equal footing
    auto compute_and_push_nbrs_eager = [&](DiskNode<T> &node, unsigned &nk) {
      unsigned nbors_cand_size = 0;
      for (unsigned m = 0; m < node.nnbrs; ++m) {
        if (visited.find(node.nbrs[m]) == visited.end()) {
          node.nbrs[nbors_cand_size++] = node.nbrs[m];
          visited.insert(node.nbrs[m]);
        }
      }
      n_computes += nbors_cand_size;
      if (nbors_cand_size) {
        nbr_handler->compute_dists(query_buf, node.nbrs, nbors_cand_size);  // PQ batch
        // Use effective_lsearch threshold — allows cold nodes to fill the extra slots.
        const float thresh = (cur_list_size == effective_lsearch)
                                 ? retset[cur_list_size - 1].distance : FLT_MAX;
        for (unsigned m = 0; m < nbors_cand_size; ++m) {
          const int nbor_id = node.nbrs[m];
          float nbor_dist = dist_scratch[m];  // PQ distance by default
          if (hot_cache_->lookup(nbor_id)) {
            // Override with exact distance for chain quality
            char *nbr_buf = hot_cache_->get_node_buf(nbor_id);
            DiskNode<T> nbr_node = node_from_page(nbr_buf, 0);
            memcpy(data_buf, nbr_node.coords, meta_.data_dim * sizeof(T));
            nbor_dist = dist_cmp->compare(query, data_buf, (unsigned)aligned_dim);
          }
          if (nbor_dist >= thresh) continue;
          if (stats != nullptr) stats->n_cmps++;
          Neighbor nn(nbor_id, nbor_dist, true);
          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
          if (cur_list_size < effective_lsearch) {
            ++cur_list_size;
            if (unlikely(cur_list_size >= retset.size())) {
              retset.resize(2 * cur_list_size);
            }
          }
          if (r < nk) nk = r;
        }
      }
    };

    auto add_to_retset = [&](const unsigned *node_ids, const uint64_t n_ids, float *dists) {
      for (uint64_t i = 0; i < n_ids; ++i) {
        retset[cur_list_size++] = Neighbor(node_ids[i], dists[i], true);
        visited.insert(node_ids[i]);
      }
    };

    // stats.
    if (stats != nullptr) {
      stats->io_us = 0;
      stats->io_us1 = 0;
      stats->cpu_us = 0;
      stats->cpu_us1 = 0;
      stats->cpu_us2 = 0;
    }
    // search in in-memory index.

    int64_t cur_beam_width = std::min(4ul, beam_width);  // before converge.
    std::vector<unsigned> mem_tags(mem_L);
    std::vector<float> mem_dists(mem_L);

#ifdef OVERLAP_INIT
    if (mem_L) {
      mem_index_->search_with_tags_fast(query, mem_L, mem_tags.data(), mem_dists.data());
      add_to_retset(mem_tags.data(), std::min((uint64_t) mem_L, l_search), mem_dists.data());
    } else {
      // cannot overlap.
      nbr_handler->initialize_query(query, query_buf);
      nbr_handler->compute_dists(query_buf, &meta_.entry_point_id, 1);
      add_to_retset(&meta_.entry_point_id, 1, dist_scratch);
    }
#else
    if (mem_L) {
      mem_index_->search_with_tags_fast(query, mem_L, mem_tags.data(), mem_dists.data());
      nbr_handler->compute_dists(query_buf, mem_tags.data(), mem_L);
      add_to_retset(mem_tags.data(), std::min((uint64_t) mem_L, l_search), dist_scratch);
    } else {
      nbr_handler->compute_dists(query_buf, &meta_.entry_point_id, 1);
      add_to_retset(&meta_.entry_point_id, 1, dist_scratch);
    }
    std::sort(retset.begin(), retset.begin() + cur_list_size);
#endif

    std::queue<io_t> on_flight_ios;
    auto send_read_req = [&](Neighbor &item) -> bool {
      item.flag = false;

      // lock the corresponding page.
      this->lock_idx(idx_lock_table, item.id, std::vector<uint32_t>(), true);
      const unsigned loc = id2loc(item.id), pid = loc_sector_no(loc);

      uint64_t &cur_buf_idx = query_buf->sector_idx;
      auto buf = sector_scratch + cur_buf_idx * size_per_io;
      auto &req = query_buf->reqs[cur_buf_idx];
      req = IORequest(static_cast<uint64_t>(pid) * SECTOR_LEN, size_per_io, buf, u_loc_offset(loc), meta_.max_node_len,
                      sector_scratch);
      reader->send_read_no_alloc(req, ctx);

      on_flight_ios.push(io_t{item, pid, loc, &req});
      cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;

      if (stats != nullptr) {
        stats->n_ios++;
      }
      return true;
    };

    std::unordered_map<unsigned, DiskNode<T>> id_buf_map;
    auto poll_all = [&]() -> std::pair<int, int> {
      auto bd_io_st = std::chrono::high_resolution_clock::now();
      // poll once.
      reader->poll_all(ctx);
      unsigned n_in = 0, n_out = 0;
      while (!on_flight_ios.empty() && on_flight_ios.front().finished()) {
        io_t &io = on_flight_ios.front();
        auto disk_node = node_from_page((char *) io.read_req->buf, io.loc);
        id_buf_map.insert(std::make_pair(io.nbr.id, disk_node));
        // Record node access for hot cache warmup
        if (this->hot_cache_ && this->hot_cache_->is_warming()) {
          this->hot_cache_->record_and_store(io.nbr.id, (char *) disk_node.coords);
        }
        io.nbr.distance <= retset[cur_list_size - 1].distance ? ++n_in : ++n_out;
        // unlock the corresponding page.
        this->unlock_idx(idx_lock_table, io.nbr.id);
        on_flight_ios.pop();
      }
      if (stats != nullptr) {
        auto bd_io_ed = std::chrono::high_resolution_clock::now();
        stats->bd_io_us += std::chrono::duration_cast<std::chrono::nanoseconds>(bd_io_ed - bd_io_st).count() / 1000.0;
      }
      return std::make_pair(n_in, n_out);
    };

    auto send_best_read_req = [&](uint32_t n) -> bool {
      // auto io_st = std::chrono::high_resolution_clock::now();
      unsigned n_sent = 0, marker = 0;
      while (marker < cur_list_size && n_sent < n) {
        while (marker < cur_list_size /* pool size */ &&
               (retset[marker].flag == false /* on flight */ ||
                id_buf_map.find(retset[marker].id) != id_buf_map.end() /* already read */)) {
          retset[marker].flag = false;  // even out the id_buf_map cost to O(1)
          ++marker;
        }
        if (marker >= cur_list_size) {
          break;  // nothing to send.
        }

        // =====================================================================
        // GateANN: Filter-Aware IO — pre-check before any IO or cache
        // =====================================================================
        // If filter_store_ is loaded and this node fails the filter:
        //   - Do NOT send SSD IO (saves 50-200μs)
        //   - Do NOT compute exact distance (saves CPU)
        //   - DO expand neighbors via full_adj_ ("tunneling") to maintain
        //     graph connectivity through filtered-out regions.
        // For single-label: filter_data points to uint8_t query_label.
        // For multi-label: filter_data points to [uint32_t count][uint32_t labels...].
        // =====================================================================
        if (this->filter_precheck_ && filter_data != nullptr &&
            ((this->filter_store_ && this->filter_store_->loaded()) ||
             (this->spmat_filter_ && this->spmat_filter_->loaded()))) {
          if (!this->check_filter(retset[marker].id, filter_data)) {
            auto bd_tunnel_st = std::chrono::high_resolution_clock::now();
            retset[marker].flag    = false;
            retset[marker].visited = true;
            // Tunnel: expand neighbors for free via full_adj_
            if (this->tunneling_loaded()) {
              uint32_t nnbrs;
              const uint32_t *nbrs;
              this->tunneling_get_adj(retset[marker].id, nnbrs, nbrs);
              uint32_t temp_nbrs[MAX_N_EDGES];
              unsigned cnt = 0;
              for (unsigned m = 0; m < nnbrs; m++) {
                if (visited.find(nbrs[m]) == visited.end()) {
                  temp_nbrs[cnt++] = nbrs[m];
                  visited.insert(nbrs[m]);
                }
              }
              if (cnt > 0) {
                nbr_handler->compute_dists(query_buf, temp_nbrs, cnt);
                for (unsigned m = 0; m < cnt; m++) {
                  float d = dist_scratch[m];
                  if (cur_list_size < effective_lsearch ||
                      d < retset[cur_list_size - 1].distance) {
                    Neighbor nn(temp_nbrs[m], d, true);
                    auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
                    if (cur_list_size < effective_lsearch) ++cur_list_size;
                    if (r < marker) marker = r;  // rescan if inserted before marker
                  }
                }
              }
            }
            if (stats != nullptr) {
              stats->n_filter_skip++;
              auto bd_tunnel_ed = std::chrono::high_resolution_clock::now();
              stats->bd_tunnel_us += std::chrono::duration_cast<std::chrono::nanoseconds>(bd_tunnel_ed - bd_tunnel_st).count() / 1000.0;
            }
            continue;  // NO IO for this node
          }
        }
        // Hot cache: serve from memory if node is cached (skip disk I/O)
        if (this->hot_cache_ && this->hot_cache_->is_populated() &&
            this->hot_cache_->lookup(retset[marker].id)) {
          retset[marker].flag = false;
          uint64_t &cur_buf_idx = query_buf->sector_idx;
          auto buf = sector_scratch + cur_buf_idx * size_per_io;
          memcpy(buf, this->hot_cache_->get_node_buf(retset[marker].id), meta_.max_node_len);
          // Lazy mode: queue for later processing by calc_best_node
          id_buf_map.insert(std::make_pair(retset[marker].id, node_from_page(buf, 0)));
          cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;
          if (stats != nullptr) stats->n_cache_hits++;
          continue;
        }
        n_sent += send_read_req(retset[marker]);
      }
      // auto io_ed = std::chrono::high_resolution_clock::now();
      // stats->io_us += std::chrono::duration_cast<std::chrono::microseconds>(io_ed - io_st).count();
      return n_sent != 0;  // nothing to send.
    };


    auto calc_best_node = [&]() -> int {  // if converged.
      auto bd_proc_st = std::chrono::high_resolution_clock::now();
      unsigned marker = 0, nk = cur_list_size, first_unvisited_eager = cur_list_size;
      /* Process one entry from id_buf_map (disk reads or lazy hot cache hits) */
      bool processed_id_buf = false;
      for (marker = 0; marker < cur_list_size; ++marker) {
        if (!retset[marker].visited && id_buf_map.find(retset[marker].id) != id_buf_map.end()) {
          retset[marker].flag = false;  // even out the id_buf_map cost to O(1)
          retset[marker].visited = true;
          auto it = id_buf_map.find(retset[marker].id);
          auto &[id, node] = *it;
          // Mode=9 early filter check: skip exact distance for non-matching nodes
          if (this->early_filter_check_ && filter_data != nullptr &&
              this->filter_store_ && this->filter_store_->loaded() &&
              !this->check_filter(id, filter_data)) {
            compute_and_push_nbrs(node, nk);
            if (stats != nullptr) stats->n_filter_skip++;
          } else {
            compute_exact_dists_and_push(node, id);
            compute_and_push_nbrs(node, nk);
          }
          processed_id_buf = true;
          break;
        }
      }


/* guess the first unvisited vector (eager) */
      for (unsigned i = 0; i < cur_list_size; ++i) {
        if (!retset[i].visited && retset[i].flag /* not on-fly */
            && id_buf_map.find(retset[i].id) == id_buf_map.end() /* not already read */) {
          first_unvisited_eager = i;
          break;
        }
      }
      if (stats != nullptr) {
        auto bd_proc_ed = std::chrono::high_resolution_clock::now();
        stats->bd_process_us += std::chrono::duration_cast<std::chrono::nanoseconds>(bd_proc_ed - bd_proc_st).count() / 1000.0;
      }
      return first_unvisited_eager;
    };

    auto get_first_unvisited = [&]() -> int {
      int ret = -1;
      for (unsigned i = 0; i < cur_list_size; ++i) {
        if (!retset[i].visited) {
          ret = i;
          break;
        }
      }
      return ret;
    };

    auto print_state = [&]() {
      LOG(INFO) << "cur_list_size: " << cur_list_size;
      for (unsigned i = 0; i < cur_list_size; ++i) {
        LOG(INFO) << "retset[" << i << "]: " << retset[i].id << ", " << retset[i].distance << ", " << retset[i].flag
                  << ", " << retset[i].visited << ", " << (id_buf_map.find(retset[i].id) != id_buf_map.end());
      }
      LOG(INFO) << "On flight IOs: " << on_flight_ios.size();
      if (on_flight_ios.size() != 0) {
        auto &io = on_flight_ios.front();
        LOG(INFO) << "on_flight_io: " << io.nbr.id << ", " << io.nbr.distance << ", " << io.nbr.flag << ", "
                  << io.page_id << ", " << io.loc << ", " << io.finished();
      }
      usleep(500);
    };

    std::ignore = print_state;

    auto cpu2_st = std::chrono::high_resolution_clock::now();
#ifdef OVERLAP_INIT
    // For filter-aware mode, PQ tables must be initialized BEFORE
    // send_best_read_req because tunneling computes PQ distances.
    // Without this, the first batch of tunneled neighbors gets garbage distances.
    if (this->filter_precheck_ && mem_L != 0) {
      nbr_handler->initialize_query(query, query_buf);
      nbr_handler->compute_dists(query_buf, mem_tags.data(), mem_L);
      for (unsigned i = 0; i < cur_list_size; ++i) {
        retset[i].distance = dist_scratch[i];
      }
      std::sort(retset.begin(), retset.begin() + cur_list_size);
    }
#endif
    send_best_read_req(cur_beam_width - on_flight_ios.size());
    unsigned marker = 0, max_marker = 0;
#ifdef OVERLAP_INIT
    if (likely(mem_L != 0) && !this->filter_precheck_) {
      nbr_handler->initialize_query(query, query_buf);
      nbr_handler->compute_dists(query_buf, mem_tags.data(), mem_L);
      for (unsigned i = 0; i < cur_list_size; ++i) {
        retset[i].distance = dist_scratch[i];
      }
      std::sort(retset.begin(), retset.begin() + cur_list_size);
    }
#endif

    int cur_n_in = 0, cur_tot = 0;
    while (get_first_unvisited() != -1) {
      // poll to heap (best-effort) -> calc best from heap (skip if heap is empty) -> send IO (if can send) -> ...
      // auto io1_st = std::chrono::high_resolution_clock::now();
      auto [n_in, n_out] = poll_all();
      std::ignore = n_in;
      std::ignore = n_out;

      if (max_marker >= 5 && n_in + n_out > 0) {
        cur_n_in += n_in;
        cur_tot += n_in + n_out;
        // converged, tune beam width.
        constexpr double kWasteThreshold = 0.1;  // 0.1 * 10
        if ((cur_tot - cur_n_in) * 1.0 / cur_tot <= kWasteThreshold) {
          cur_beam_width = cur_beam_width + 1;
          cur_beam_width = std::max(cur_beam_width, 4l);
          cur_beam_width = std::min((int64_t) beam_width, cur_beam_width);
        }

        // In ensure_enough_results mode: record converge_size when first converged
        if (relaxed_monotonicity_l > 0 && converge_size < 0) {
          converge_size = full_retset.size();
        }
      }

      // In relaxed_monotonicity mode: terminate when full_retset.size() >= converge_size + relaxed_monotonicity_l
      if (relaxed_monotonicity_l && converge_size >= 0 &&
          full_retset.size() >= static_cast<size_t>(converge_size) + relaxed_monotonicity_l) {
        break;
      }

      if ((int64_t) on_flight_ios.size() < cur_beam_width) {
        send_best_read_req(1);
      }
      // auto io1_ed = std::chrono::high_resolution_clock::now();
      // stats->io_us1 += std::chrono::duration_cast<std::chrono::microseconds>(io1_ed - io1_st).count();
      marker = calc_best_node();
      max_marker = std::max(max_marker, marker);
    }

    // In relaxed_monotonicity mode: drain all on-flight IOs and process remaining nodes
    if (relaxed_monotonicity_l > 0) {
      while (!on_flight_ios.empty()) {
        reader->poll_all(ctx);
        while (!on_flight_ios.empty() && on_flight_ios.front().finished()) {
          io_t &io = on_flight_ios.front();
          auto drain_node = node_from_page((char *) io.read_req->buf, io.loc);
          id_buf_map.insert(std::make_pair(io.nbr.id, drain_node));
          if (this->hot_cache_ && this->hot_cache_->is_warming()) {
            this->hot_cache_->record_and_store(io.nbr.id, (char *) drain_node.coords);
          }
          this->unlock_idx(idx_lock_table, io.nbr.id);
          on_flight_ios.pop();
        }
      }
      // Process remaining unvisited nodes in id_buf_map to add to full_retset
      for (unsigned i = 0; i < cur_list_size; ++i) {
        if (!retset[i].visited && id_buf_map.find(retset[i].id) != id_buf_map.end()) {
          retset[i].visited = true;
          auto it = id_buf_map.find(retset[i].id);
          auto &[id, node] = *it;
          compute_exact_dists_and_push(node, id);
        }
      }
    }

    auto cpu2_ed = std::chrono::high_resolution_clock::now();
    if (stats != nullptr) {
      stats->cpu_us2 = std::chrono::duration_cast<std::chrono::microseconds>(cpu2_ed - cpu2_st).count();
      stats->cpu_us = n_computes;
    }
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    // copy k_search values
    uint64_t t = 0;
    for (uint64_t i = 0; i < full_retset.size() && t < k_search; i++) {
      if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
        continue;  // deduplicate.
      }
      res_tags[t] = id2tag(full_retset[i].id);
      if (distances != nullptr) {
        distances[t] = full_retset[i].distance;
      }
      t++;
    }

    push_query_buf(query_buf);

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
    return t;
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
