#include <index.h>
#include <omp.h>
#include <string.h>
#include <numeric>
#include "utils.h"
#include <sys/mman.h>
#include <unistd.h>

template<typename T>
int build_in_memory_index(const std::string &data_path, const std::string &tags_file, const unsigned R,
                          const unsigned L, const float alpha, const std::string &save_path, const unsigned num_threads,
                          pipeann::Metric distMetric) {
  pipeann::IndexBuildParameters paras;
  paras.set(R, L, 750, alpha, num_threads, false);

  uint64_t data_num, data_dim;
  pipeann::get_bin_metadata(data_path, data_num, data_dim);
  std::cout << "Building in-memory index with parameters: data_file: " << data_path << "tags file: " << tags_file
            << " R: " << R << " L: " << L << " alpha: " << alpha << " index_path: " << save_path
            << " #threads: " << num_threads
            << ", using distance metric: " << (distMetric == pipeann::Metric::COSINE ? "cosine " : "l2 ");

  typedef uint32_t TagT;

  pipeann::Index<T, TagT> index(distMetric, data_dim);
  std::cout << "Opening bin file " << tags_file << "... " << std::endl;
  std::vector<TagT> tags;
  pipeann::load_bin<TagT>(tags_file, tags, data_num, data_dim);

  auto s = std::chrono::high_resolution_clock::now();
  index.build(data_path.c_str(), data_num, paras, tags);
  std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

  std::cout << "Indexing time: " << diff.count() << "\n";

  index.save(save_path.c_str());

  return 0;
}

int main(int argc, char **argv) {
  if (argc != 10) {
    std::cout << "Usage: " << argv[0]
              << " <data_type(int8/uint8/float)>  <data_file.bin> <tags_file>"
                 " <output_index_file>"
              << " <R> <L> <alpha> <num_threads_to_use>"
              << " <distance_metric(l2/cosine/mips)>."
              << " See README for more information on parameters." << std::endl;
    exit(-1);
  }

  int arg_no = 2;

  const std::string data_path(argv[arg_no++]);
  const std::string tags_file(argv[arg_no++]);
  const std::string save_path(argv[arg_no++]);
  const unsigned R = (unsigned) atoi(argv[arg_no++]);
  const unsigned L = (unsigned) atoi(argv[arg_no++]);
  const float alpha = (float) atof(argv[arg_no++]);
  const unsigned num_threads = (unsigned) atoi(argv[arg_no++]);
  const std::string dist_metric_str = argv[arg_no++];
  pipeann::Metric distMetric = pipeann::get_metric(dist_metric_str);

  if (std::string(argv[1]) == std::string("int8"))
    build_in_memory_index<int8_t>(data_path, tags_file, R, L, alpha, save_path, num_threads, distMetric);
  else if (std::string(argv[1]) == std::string("uint8"))
    build_in_memory_index<uint8_t>(data_path, tags_file, R, L, alpha, save_path, num_threads, distMetric);
  else if (std::string(argv[1]) == std::string("float"))
    build_in_memory_index<float>(data_path, tags_file, R, L, alpha, save_path, num_threads, distMetric);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}