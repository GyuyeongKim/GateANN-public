// =============================================================================
// GateANN: Filter-Aware Pipe Search Experiment Binary
// =============================================================================
// Compares two modes on filtered ANNS workloads:
//   mode=4 (baseline): send IO for ALL nodes, post-filter results
//   mode=8 (ours):     pre-IO filter check + tunneling via FullAdjIndex
//
// Usage:
//   ./search_disk_index_fa <type(float/uint8)> <index_prefix>
//       <num_threads> <beamwidth>
//       <query_bin> <node_labels_bin> <query_labels_bin> <filtered_gt_bin>
//       <K> <dist_metric> <nbr_type>
//       <mode(4=baseline,8=filter-aware)>
//       <mem_L> <cache_budget>
//       [full_adj_max_nbrs (mode=8, default=32)]
//       <L1> [L2] ...
// =============================================================================

#include <omp.h>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

#include "ssd_index.h"

// Read current process RSS from /proc/self/status (Linux only)
static double get_rss_gb() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line)) {
    if (line.substr(0, 6) == "VmRSS:") {
      // Format: "VmRSS:    12345 kB"
      long kb = 0;
      sscanf(line.c_str(), "VmRSS: %ld", &kb);
      return kb / (1024.0 * 1024.0);
    }
  }
  return 0;
}
#include "utils/log.h"
#include "utils/timer.h"
#include "utils.h"
#include "linux_aligned_file_reader.h"
#include "filter/selector.h"

// Simple label-equality selector for post-filter baseline (mode=4).
// node.labels is not used — we look up filter_store directly via filter_data.
struct LabelEqSelector : pipeann::AbstractSelector {
  const pipeann::FilterStore *fs;
  explicit LabelEqSelector(const pipeann::FilterStore *fs_) : fs(fs_) {}
  bool is_member(uint32_t target_id, const void *query_labels, const void *) override {
    if (!fs || !query_labels) return true;
    uint8_t ql = *(const uint8_t *)query_labels;
    return fs->passes(target_id, ql);
  }
};

// Load query labels from binary: [uint32 n][uint8 label[n]]
std::vector<uint8_t> load_labels(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open " + path);
  uint32_t n;
  f.read((char *)&n, sizeof(n));
  std::vector<uint8_t> labels(n);
  f.read((char *)labels.data(), n);
  return labels;
}

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
  for (uint64_t i = 0; i < query_num; i++) {
    std::set<unsigned> gt_set(gt + i * gt_dim,
                               gt + i * gt_dim + std::min(gt_dim, recall_at));
    unsigned hits = 0;
    for (uint64_t j = 0; j < recall_at; j++) {
      if (gt_set.count(res[i * recall_at + j])) hits++;
    }
    total += (double)hits / (double)std::min(gt_dim, recall_at);
  }
  return total / query_num;
}

template<typename T>
int run(int argc, char **argv) {
  int idx = 2;
  std::string index_prefix(argv[idx++]);
  uint32_t num_threads = std::atoi(argv[idx++]);
  uint32_t beamwidth   = std::atoi(argv[idx++]);
  std::string query_bin      = argv[idx++];
  std::string node_labels_bin= argv[idx++];
  std::string qlabels_bin    = argv[idx++];
  std::string gt_bin         = argv[idx++];
  uint64_t recall_at   = std::atoi(argv[idx++]);
  std::string dist_metric    = argv[idx++];
  std::string nbr_type       = argv[idx++];
  int search_mode      = std::atoi(argv[idx++]);
  uint32_t mem_L       = std::atoi(argv[idx++]);
  uint32_t cache_budget= std::atoi(argv[idx++]);
  uint32_t full_adj_nbrs = 32;
  if (search_mode == FILTER_AWARE_PIPE_SEARCH && idx < argc - 1 &&
      std::string(argv[idx]).find_first_not_of("0123456789") == std::string::npos &&
      std::stoi(argv[idx]) < 200) {
    full_adj_nbrs = std::atoi(argv[idx++]);
  }

  // Parse optional flags: --lazy_adj
  bool lazy_adj = false;
  while (idx < argc && std::string(argv[idx]).substr(0, 2) == "--") {
    std::string flag(argv[idx++]);
    if (flag == "--lazy_adj") lazy_adj = true;
    else { std::cerr << "Unknown flag: " << flag << "\n"; return -1; }
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

  auto node_labels  = load_labels(node_labels_bin);
  auto query_labels = load_labels(qlabels_bin);
  if (query_labels.size() < query_num) {
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

  // mode=0 and mode=10 use synchronous beam search; others use async IO
  bool use_page_search = (search_mode != BEAM_SEARCH && search_mode != 10);
  int res = idx_->load(index_prefix.c_str(), num_threads, use_page_search);
  if (res != 0) return res;

  if (mem_L != 0) idx_->load_mem_index(index_prefix + "_mem.index");
  omp_set_num_threads(num_threads);

  // Load filter store (always — used for both mode=4 post-filter and mode=8 pre-filter)
  idx_->load_filter_store(node_labels_bin);

  // Cache warmup (mode=4 and mode=8 both use hot cache for QPS)
  if (cache_budget > 0) {
    idx_->init_cache(cache_budget);
    std::cout << "Warmup: " << query_num << " queries, L=" << Lvec[0] << " ..." << std::flush;
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t)query_num; i++) {
      std::vector<uint32_t> tags(recall_at);
      std::vector<float>    dists(recall_at);
      pipeann::QueryStats   dummy;
      uint8_t ql = query_labels[i];
      // Warmup without filter so hot nodes reflect full graph traversal
      idx_->pipe_search(query + i * query_dim, recall_at, mem_L, Lvec[0],
                        tags.data(), dists.data(), beamwidth, &dummy);
    }
    idx_->finalize_cache();
    std::cout << " done. Cache: " << idx_->get_cache()->size() << " nodes\n";
  }

  // Build FullAdjIndex and enable pre-IO filter check for mode=8 only
  if (search_mode == FILTER_AWARE_PIPE_SEARCH) {
    if (lazy_adj) std::cout << "FullAdjIndex: lazy mmap (on-demand paging)" << std::endl;
    idx_->build_full_adj(full_adj_nbrs, lazy_adj);
    idx_->set_filter_precheck(true);
    std::cout << "RSS after FullAdjIndex load: "
              << std::fixed << std::setprecision(2) << get_rss_gb() << " GB" << std::endl;
  }

  // mode=9: Early Filter Check — IO still happens, skip exact dist for non-matching
  if (search_mode == EARLY_FILTER_PIPE_SEARCH) {
    idx_->set_early_filter_check(true);
    std::cout << "Early Filter Check mode: IO unchanged, skip exact distance for non-matching nodes" << std::endl;
  }

  // mode=10: Filtered-DiskANN — hard filter during candidate expansion.
  // Official approach: compute PQ distances for all neighbors, then only add
  // filter-matching neighbors to the candidate set.  Non-matching nodes are
  // dropped, so they never get expanded.  All disk IOs still happen (no pre-IO
  // filter skip), which is the key difference from GateANN (mode=8).
  if (search_mode == 10) {
    idx_->set_fdiskann_filter(true);
    idx_->compute_fdiskann_medoids();
    std::cout << "Filtered-DiskANN mode: hard candidate filter + per-label medoids" << std::endl;
  }

  // Selector for mode=4 post-filter baseline
  LabelEqSelector selector(idx_->get_filter_store());

  // Result buffers
  std::vector<std::vector<uint32_t>> results(
      Lvec.size(), std::vector<uint32_t>(query_num * recall_at, 0));
  std::vector<pipeann::QueryStats> stats(query_num);

  // Print header
  bool is_fa = (search_mode == FILTER_AWARE_PIPE_SEARCH);
  bool is_early = (search_mode == EARLY_FILTER_PIPE_SEARCH);
  bool is_beam = (search_mode == BEAM_SEARCH);
  bool is_fdiskann = (search_mode == 10);
  std::string mode_name = is_fa ? "FilterAware(mode=8)"
                        : is_early ? "EarlyFilter(mode=9)"
                        : is_beam ? "DiskANN(mode=0)"
                        : is_fdiskann ? "FilteredDiskANN(mode=10)"
                        : "Baseline(mode=2)";
  std::cout << "\n=== " << mode_name << " ===\n";
  std::cout << std::setw(6) << "L"
            << std::setw(12) << "QPS"
            << std::setw(12) << "Recall@" + std::to_string(recall_at)
            << std::setw(12) << "MeanIOs"
            << std::setw(14) << "FilterSkips"
            << std::setw(12) << "IO_us"
            << std::setw(12) << "Tunnel_us"
            << std::setw(12) << "Process_us"
            << std::setw(10) << "RSS_GB"
            << "\n";
  std::cout << std::string(106, '-') << std::endl;

  for (size_t test_id = 0; test_id < Lvec.size(); test_id++) {
    uint64_t L = Lvec[test_id];
    auto &res_buf = results[test_id];
    std::fill(stats.begin(), stats.end(), pipeann::QueryStats{});

    auto t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t)query_num; i++) {
      uint8_t ql = query_labels[i];  // per-query filter label
      if (is_fa || is_early) {
        // mode=8: pre-IO filter + tunneling
        // mode=9: early filter check (IO happens, skip exact dist for non-matching)
        idx_->pipe_search(query + i * query_dim, recall_at, mem_L, L,
                          res_buf.data() + i * recall_at,
                          nullptr, beamwidth, &stats[i],
                          nullptr, &ql);
      } else if (is_beam || is_fdiskann) {
        // mode=0: DiskANN beam_search + post-filter
        // mode=10: Filtered-DiskANN beam_search + hard candidate filter
        // Both use synchronous IO; mode=10 additionally passes filter_data
        // to beam_search so non-matching neighbors are skipped during expansion.
        std::vector<uint32_t> cand_tags(L);
        std::vector<float> cand_dists(L);
        idx_->beam_search(query + i * query_dim, L, mem_L, L,
                          cand_tags.data(), cand_dists.data(),
                          beamwidth, &stats[i],
                          nullptr, false,
                          is_fdiskann ? (const void *)&ql : nullptr);
        // Post-filter: keep only nodes with matching label
        uint32_t *out = res_buf.data() + i * recall_at;
        size_t found = 0;
        for (size_t j = 0; j < L && found < recall_at; j++) {
          if (selector.is_member(cand_tags[j], &ql, nullptr)) {
            out[found++] = cand_tags[j];
          }
        }
      } else {
        // mode=2/4: PipeANN pipe_search + post-filter via selector
        idx_->pipe_search(query + i * query_dim, recall_at, mem_L, L,
                          res_buf.data() + i * recall_at,
                          nullptr, beamwidth, &stats[i],
                          &selector, &ql);
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
              << std::setw(10) << std::fixed << std::setprecision(2) << get_rss_gb()
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
              << " <query.fbin> <node_labels.bin> <query_labels.bin> <filtered_gt.bin>"
              << " <K> <metric> <nbr_type>"
              << " <mode(0=diskann,2=pipeann,8=filter-aware,10=filtered-diskann)>"
              << " <mem_L> <cache_budget>"
              << " [full_adj_max_nbrs(mode=8)]"
              << " [--lazy_adj]"
              << " <L1> [L2] ...\n";
    return -1;
  }
  std::string type = argv[1];
  if (type == "float") return run<float>(argc, argv);
  if (type == "uint8") return run<uint8_t>(argc, argv);
  std::cerr << "Unknown type: " << type << "\n";
  return -1;
}
