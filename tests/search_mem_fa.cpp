// =============================================================================
// In-Memory Vamana Search with Post-Filter
// =============================================================================
// Loads the disk index fully into RAM (like search_disk_index_mem) and runs
// in-memory greedy search with post-filtering.
// Used to measure "ideal" in-memory performance for comparison.
//
// Usage:
//   ./search_mem_fa <type(float/uint8)> <index_prefix>
//       <num_threads> <query_bin>
//       <node_labels_bin> <query_labels_bin> <filtered_gt_bin>
//       <K> <dist_metric>
//       <L1> [L2] ...
// =============================================================================

#include <omp.h>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <chrono>

#include "ssd_index.h"
#include "index.h"
#include "nbr/dummy_nbr.h"
#include "utils/log.h"
#include "utils.h"
#include "linux_aligned_file_reader.h"

// Load labels from binary: [uint32 n][uint8 label[n]]
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
  uint64_t k = std::min(gt_dim, recall_at);
  if (k == 0) return 0.0;
  for (uint64_t i = 0; i < query_num; i++) {
    std::set<unsigned> gt_set(gt + i * gt_dim, gt + i * gt_dim + k);
    unsigned hits = 0;
    for (uint64_t j = 0; j < recall_at; j++) {
      if (gt_set.count(res[i * recall_at + j])) hits++;
    }
    total += (double)hits / (double)k;
  }
  return total / query_num;
}

template<typename T>
int run(int argc, char **argv) {
  int idx = 2;
  std::string index_prefix(argv[idx++]);
  uint32_t num_threads = std::atoi(argv[idx++]);
  std::string query_bin      = argv[idx++];
  std::string node_labels_bin= argv[idx++];
  std::string qlabels_bin    = argv[idx++];
  std::string gt_bin         = argv[idx++];
  uint64_t recall_at   = std::atoi(argv[idx++]);
  std::string dist_metric    = argv[idx++];

  std::vector<uint64_t> Lvec;
  for (int c = idx; c < argc; c++) {
    uint64_t L = std::atoi(argv[c]);
    if (L >= recall_at) Lvec.push_back(L);
  }
  if (Lvec.empty()) { std::cerr << "No valid L values\n"; return -1; }

  // Load query data
  T *query = nullptr;
  size_t query_num, query_dim;
  pipeann::load_bin<T>(query_bin, query, query_num, query_dim);
  std::cout << "Loaded " << query_num << " queries, dim=" << query_dim << std::endl;

  auto query_labels = load_labels(qlabels_bin);
  auto node_labels = load_labels(node_labels_bin);
  std::cout << "Loaded " << node_labels.size() << " node labels, "
            << query_labels.size() << " query labels" << std::endl;

  unsigned *gt_ids = nullptr;
  uint64_t gt_num, gt_dim;
  load_filtered_gt(gt_bin, gt_ids, gt_num, gt_dim);
  std::cout << "Loaded GT: " << gt_num << " queries, K=" << gt_dim << std::endl;

  // Load disk index into memory using DummyNeighbor (same as search_disk_index_mem)
  pipeann::Metric m = pipeann::get_metric(dist_metric);
  std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  pipeann::SSDIndex<T> ssd_idx(m, reader, new pipeann::DummyNeighbor<T>(m), true, nullptr);

  std::cout << "Loading disk index into memory..." << std::flush;
  auto t_load_start = std::chrono::high_resolution_clock::now();
  auto &mem_idx = *ssd_idx.load_to_mem(index_prefix);
  auto t_load_end = std::chrono::high_resolution_clock::now();
  double load_sec = std::chrono::duration<double>(t_load_end - t_load_start).count();
  std::cout << " done. (" << std::fixed << std::setprecision(1) << load_sec << "s)" << std::endl;

  omp_set_num_threads(num_threads);

  // Print header
  std::cout << "\n=== In-Memory Vamana (post-filter) ===" << std::endl;
  std::cout << std::setw(6) << "L"
            << std::setw(12) << "QPS"
            << std::setw(12) << "Recall@" + std::to_string(recall_at)
            << std::setw(12) << "MeanIOs"
            << std::setw(14) << "FilterSkips"
            << std::endl;
  std::cout << std::string(56, '-') << std::endl;

  for (size_t test_id = 0; test_id < Lvec.size(); test_id++) {
    uint64_t L = Lvec[test_id];

    std::vector<uint32_t> res_buf(query_num * recall_at, 0);
    pipeann::QueryStats *stats = new pipeann::QueryStats[query_num];

    auto t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t)query_num; i++) {
      uint8_t ql = query_labels[i];

      // In-memory greedy search — retrieve L candidates (not just K)
      // so post-filter has enough candidates to find recall_at matches
      std::vector<uint32_t> cand_tags(L);
      std::vector<float> cand_dists(L);
      mem_idx.search(query + i * query_dim, L, L,
                     cand_tags.data(), cand_dists.data(), &stats[i]);

      // Post-filter: keep matching labels
      uint32_t *out = res_buf.data() + i * recall_at;
      size_t found = 0;
      for (size_t j = 0; j < L && found < recall_at; j++) {
        uint32_t tag = cand_tags[j];
        if (tag < node_labels.size() && node_labels[tag] == ql) {
          out[found++] = tag;
        }
      }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

    double qps = query_num / elapsed_s;
    double recall = compute_recall(gt_ids, gt_dim, res_buf.data(),
                                   query_num, recall_at);

    float mean_ios = (float)pipeann::get_mean_stats(
        stats, query_num, [](const pipeann::QueryStats &s) { return s.n_ios; });

    std::cout << std::setw(6) << L
              << std::setw(12) << std::fixed << std::setprecision(0) << qps
              << std::setw(12) << std::fixed << std::setprecision(4) << recall
              << std::setw(12) << std::fixed << std::setprecision(2) << mean_ios
              << std::setw(14) << std::fixed << std::setprecision(2) << 0.0
              << std::endl;

    delete[] stats;
  }

  delete[] query;
  delete[] gt_ids;
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 11) {
    std::cerr << "Usage: " << argv[0]
              << " <float|uint8> <index_prefix> <threads>"
              << " <query.bin> <node_labels.bin> <query_labels.bin> <filtered_gt.bin>"
              << " <K> <metric(l2/cosine)>"
              << " <L1> [L2] ...\n";
    return -1;
  }
  std::string type = argv[1];
  if (type == "float") return run<float>(argc, argv);
  if (type == "uint8") return run<uint8_t>(argc, argv);
  std::cerr << "Unknown type: " << type << "\n";
  return -1;
}
