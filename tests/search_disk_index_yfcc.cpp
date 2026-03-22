// =============================================================================
// YFCC-10M Multi-Label Filtered Search Experiment
// =============================================================================
// Supports post-filter (mode=2) and pre-IO filter + tunneling (mode=8)
// with multi-label subset predicate (query tags ⊆ base tags).
//
// Usage:
//   ./search_disk_index_yfcc <type(uint8)> <index_prefix>
//       <num_threads> <beamwidth>
//       <query_bin> <base_metadata.spmat> <query_metadata.spmat> <filtered_gt.bin>
//       <K> <dist_metric> <nbr_type>
//       <mode(2=pipeann,8=gateann)>
//       <mem_L> <cache_budget>
//       [full_adj_max_nbrs (mode=8, default=32)]
//       <L1> [L2] ...
// =============================================================================

#include <omp.h>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>

#include "ssd_index.h"
#include "utils/log.h"
#include "utils/timer.h"
#include "utils.h"
#include "linux_aligned_file_reader.h"
#include "filter/selector.h"

// Multi-label post-filter selector using SpmatFilterStore in DRAM.
struct SpmatPostFilterSelector : pipeann::AbstractSelector {
  const pipeann::SpmatFilterStore *sf;
  explicit SpmatPostFilterSelector(const pipeann::SpmatFilterStore *sf_) : sf(sf_) {}
  bool is_member(uint32_t target_id, const void *query_labels, const void *) override {
    if (!sf || !query_labels) return true;
    return sf->passes(target_id, query_labels);
  }
};

// Load spmat and extract per-query label buffers.
// Returns vector of buffers, each formatted as [uint32 count][uint32 label1]...[uint32 labelN]
struct QueryLabels {
  std::vector<std::vector<uint32_t>> buffers;  // per-query [count, label1, ..., labelN]

  void load(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open " + path);

    int64_t nrow, ncol, nnz;
    in.read((char *)&nrow, sizeof(int64_t));
    in.read((char *)&ncol, sizeof(int64_t));
    in.read((char *)&nnz, sizeof(int64_t));

    std::vector<int64_t> indptr(nrow + 1);
    in.read((char *)indptr.data(), (nrow + 1) * sizeof(int64_t));
    std::vector<int32_t> indices(nnz);
    in.read((char *)indices.data(), nnz * sizeof(int32_t));
    std::vector<float> data(nnz);
    in.read((char *)data.data(), nnz * sizeof(float));
    in.close();

    buffers.resize(nrow);
    for (int64_t i = 0; i < nrow; i++) {
      std::vector<uint32_t> labels;
      for (int64_t j = indptr[i]; j < indptr[i + 1]; j++) {
        if (data[j] != 0.0f) {
          labels.push_back((uint32_t)indices[j]);
        }
      }
      std::sort(labels.begin(), labels.end());  // must be sorted for subset check
      // Format: [count, label1, ..., labelN]
      buffers[i].resize(1 + labels.size());
      buffers[i][0] = (uint32_t)labels.size();
      for (size_t k = 0; k < labels.size(); k++) {
        buffers[i][1 + k] = labels[k];
      }
    }
    std::cout << "QueryLabels loaded: " << nrow << " queries" << std::endl;
  }

  const void *get(size_t qi) const {
    return (const void *)buffers[qi].data();
  }
};

// Load filtered GT: [uint32 nq][uint32 K][int32 gt[nq*K]]
void load_filtered_gt(const std::string &path,
                      unsigned *&gt_ids, uint64_t &gt_num, uint64_t &gt_dim) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open GT " + path);
  uint32_t nq, K;
  f.read((char *)&nq, sizeof(nq));
  f.read((char *)&K,  sizeof(K));
  gt_num = nq; gt_dim = K;
  gt_ids = new unsigned[nq * K];
  std::vector<int32_t> tmp(nq * K);
  f.read((char *)tmp.data(), nq * K * sizeof(int32_t));
  for (size_t i = 0; i < (size_t)nq * K; i++)
    gt_ids[i] = (unsigned)tmp[i];
}

double compute_recall(unsigned *gt, uint64_t gt_dim, uint32_t *res,
                      uint64_t query_num, uint64_t recall_at) {
  double total = 0;
  uint64_t valid_queries = 0;
  for (uint64_t i = 0; i < query_num; i++) {
    // Skip queries with no valid GT (all -1)
    uint64_t valid_gt = 0;
    for (uint64_t j = 0; j < std::min(gt_dim, recall_at); j++) {
      if (gt[i * gt_dim + j] != (unsigned)-1) valid_gt++;
    }
    if (valid_gt == 0) continue;
    valid_queries++;

    std::set<unsigned> gt_set;
    for (uint64_t j = 0; j < std::min(gt_dim, recall_at); j++) {
      if (gt[i * gt_dim + j] != (unsigned)-1)
        gt_set.insert(gt[i * gt_dim + j]);
    }
    unsigned hits = 0;
    for (uint64_t j = 0; j < recall_at; j++) {
      if (gt_set.count(res[i * recall_at + j])) hits++;
    }
    total += (double)hits / (double)gt_set.size();
  }
  return valid_queries > 0 ? total / valid_queries : 0.0;
}

template<typename T>
int run(int argc, char **argv) {
  int idx = 2;
  std::string index_prefix(argv[idx++]);
  uint32_t num_threads = std::atoi(argv[idx++]);
  uint32_t beamwidth   = std::atoi(argv[idx++]);
  std::string query_bin       = argv[idx++];
  std::string base_meta_path  = argv[idx++];
  std::string query_meta_path = argv[idx++];
  std::string gt_bin          = argv[idx++];
  uint64_t recall_at   = std::atoi(argv[idx++]);
  std::string dist_metric     = argv[idx++];
  std::string nbr_type        = argv[idx++];
  int search_mode      = std::atoi(argv[idx++]);
  uint32_t mem_L       = std::atoi(argv[idx++]);
  uint32_t cache_budget= std::atoi(argv[idx++]);
  uint32_t full_adj_nbrs = 32;
  if (search_mode == FILTER_AWARE_PIPE_SEARCH && idx < argc - 1 &&
      std::string(argv[idx]).find_first_not_of("0123456789") == std::string::npos &&
      std::stoi(argv[idx]) < 200) {
    full_adj_nbrs = std::atoi(argv[idx++]);
  }

  std::vector<uint64_t> Lvec;
  for (int c = idx; c < argc; c++) {
    uint64_t L = std::atoi(argv[c]);
    if (L >= recall_at) Lvec.push_back(L);
  }
  if (Lvec.empty()) { std::cerr << "No valid L values\n"; return -1; }

  // Load data
  T *query = nullptr;
  size_t query_num, query_dim;
  pipeann::load_bin<T>(query_bin, query, query_num, query_dim);

  // Load multi-label metadata
  QueryLabels query_labels;
  query_labels.load(query_meta_path);
  if (query_labels.buffers.size() < query_num) {
    std::cerr << "query_labels size mismatch\n"; return -1;
  }

  unsigned *gt_ids = nullptr;
  uint64_t gt_num, gt_dim;
  load_filtered_gt(gt_bin, gt_ids, gt_num, gt_dim);

  // Load index
  pipeann::Metric m = pipeann::get_metric(dist_metric);
  auto *nbr_handler = pipeann::get_nbr_handler<T>(m, nbr_type);
  std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  std::unique_ptr<pipeann::SSDIndex<T>> idx_(
      new pipeann::SSDIndex<T>(m, reader, nbr_handler, true));

  bool use_page_search = (search_mode != BEAM_SEARCH);
  int res = idx_->load(index_prefix.c_str(), num_threads, use_page_search);
  if (res != 0) return res;

  if (mem_L != 0) idx_->load_mem_index(index_prefix + "_mem.index");
  omp_set_num_threads(num_threads);

  // Load SpmatFilterStore (base metadata in DRAM)
  idx_->load_spmat_filter_store(base_meta_path);

  // Cache warmup
  if (cache_budget > 0) {
    idx_->init_cache(cache_budget);
    std::cout << "Warmup: " << query_num << " queries, L=" << Lvec[0] << " ..." << std::flush;
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t)query_num; i++) {
      std::vector<uint32_t> tags(recall_at);
      std::vector<float>    dists(recall_at);
      pipeann::QueryStats   dummy;
      idx_->pipe_search(query + i * query_dim, recall_at, mem_L, Lvec[0],
                        tags.data(), dists.data(), beamwidth, &dummy);
    }
    idx_->finalize_cache();
    std::cout << " done. Cache: " << idx_->get_cache()->size() << " nodes\n";
  }

  // Build FullAdjIndex and enable pre-IO filter check for mode=8
  if (search_mode == FILTER_AWARE_PIPE_SEARCH) {
    idx_->build_full_adj(full_adj_nbrs);
    idx_->set_filter_precheck(true);
  }

  // Selector for mode=2 post-filter baseline
  SpmatPostFilterSelector selector(idx_->get_spmat_filter());

  // Result buffers
  std::vector<std::vector<uint32_t>> results(
      Lvec.size(), std::vector<uint32_t>(query_num * recall_at, 0));
  std::vector<pipeann::QueryStats> stats(query_num);

  bool is_fa = (search_mode == FILTER_AWARE_PIPE_SEARCH);
  std::string mode_name = is_fa ? "GateANN(mode=8)" : "PipeANN(mode=2)";
  std::cout << "\n=== " << mode_name << " [YFCC multi-label] ===\n";
  std::cout << std::setw(6) << "L"
            << std::setw(12) << "QPS"
            << std::setw(12) << "Recall@" + std::to_string(recall_at)
            << std::setw(12) << "MeanIOs"
            << std::setw(14) << "FilterSkips"
            << std::setw(12) << "IO_us"
            << std::setw(12) << "Tunnel_us"
            << std::setw(12) << "Process_us"
            << "\n";
  std::cout << std::string(96, '-') << std::endl;

  for (size_t test_id = 0; test_id < Lvec.size(); test_id++) {
    uint64_t L = Lvec[test_id];
    auto &res_buf = results[test_id];
    std::fill(stats.begin(), stats.end(), pipeann::QueryStats{});

    auto t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t)query_num; i++) {
      const void *ql = query_labels.get(i);
      if (is_fa) {
        // mode=8: pre-IO filter + tunneling
        idx_->pipe_search(query + i * query_dim, recall_at, mem_L, L,
                          res_buf.data() + i * recall_at,
                          nullptr, beamwidth, &stats[i],
                          nullptr, ql);
      } else {
        // mode=2: PipeANN pipe_search + post-filter via selector
        idx_->pipe_search(query + i * query_dim, recall_at, mem_L, L,
                          res_buf.data() + i * recall_at,
                          nullptr, beamwidth, &stats[i],
                          &selector, ql);
      }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

    double qps    = query_num / elapsed_s;
    double recall = compute_recall(gt_ids, gt_dim, res_buf.data(),
                                   query_num, recall_at);
    double mean_ios = pipeann::get_mean_stats(
        stats.data(), query_num,
        [](const pipeann::QueryStats &s) { return s.n_ios; });
    double mean_filter_skip = pipeann::get_mean_stats(
        stats.data(), query_num,
        [](const pipeann::QueryStats &s) { return s.n_filter_skip; });
    double mean_bd_io = pipeann::get_mean_stats(
        stats.data(), query_num,
        [](const pipeann::QueryStats &s) { return s.bd_io_us; });
    double mean_bd_tunnel = pipeann::get_mean_stats(
        stats.data(), query_num,
        [](const pipeann::QueryStats &s) { return s.bd_tunnel_us; });
    double mean_bd_process = pipeann::get_mean_stats(
        stats.data(), query_num,
        [](const pipeann::QueryStats &s) { return s.bd_process_us; });

    std::cout << std::setw(6)  << L
              << std::setw(12) << std::fixed << std::setprecision(0) << qps
              << std::setw(12) << std::fixed << std::setprecision(4) << recall
              << std::setw(12) << std::fixed << std::setprecision(2) << mean_ios
              << std::setw(14) << std::fixed << std::setprecision(2) << mean_filter_skip
              << std::setw(12) << std::fixed << std::setprecision(0) << mean_bd_io
              << std::setw(12) << std::fixed << std::setprecision(0) << mean_bd_tunnel
              << std::setw(12) << std::fixed << std::setprecision(0) << mean_bd_process
              << std::endl;
  }

  delete[] query;
  delete[] gt_ids;
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 16) {
    std::cerr << "Usage: " << argv[0]
              << " <float|uint8> <index_prefix> <threads> <bw>"
              << " <query.u8bin> <base_metadata.spmat> <query_metadata.spmat> <filtered_gt.bin>"
              << " <K> <metric> <nbr_type>"
              << " <mode(2=pipeann,8=gateann)>"
              << " <mem_L> <cache_budget>"
              << " [full_adj_max_nbrs(mode=8)]"
              << " <L1> [L2] ...\n";
    return -1;
  }
  std::string type = argv[1];
  if (type == "float") return run<float>(argc, argv);
  if (type == "uint8") return run<uint8_t>(argc, argv);
  std::cerr << "Unknown type: " << type << "\n";
  return -1;
}
