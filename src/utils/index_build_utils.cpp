#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cblas.h>
#include <chrono>
#include <cblas.h>

#include "utils/index_build_utils.h"
#include "utils/cached_io.h"
#include "index.h"
#include "omp.h"
#include "utils/partition.h"
#include "utils.h"

namespace pipeann {
  template<typename T>
  void normalize_data_file(const std::string &inFileName, const std::string &outFileName) {
    std::ifstream readr(inFileName, std::ios::binary);
    std::ofstream writr(outFileName, std::ios::binary);

    int npts_s32, ndims_s32;
    readr.read((char *) &npts_s32, sizeof(int32_t));
    readr.read((char *) &ndims_s32, sizeof(int32_t));

    writr.write((char *) &npts_s32, sizeof(int32_t));
    writr.write((char *) &ndims_s32, sizeof(int32_t));

    uint64_t npts = (uint64_t) npts_s32, ndims = (uint64_t) ndims_s32;
    LOG(INFO) << "Normalizing vectors in file: " << inFileName;
    LOG(INFO) << "Dataset: #pts = " << npts << ", # dims = " << ndims;

    uint64_t blk_size = 131072;
    uint64_t nblks = ROUND_UP(npts, blk_size) / blk_size;
    LOG(INFO) << "# blks: " << nblks;

    T *read_buf = new T[blk_size * ndims];
    for (uint64_t i = 0; i < nblks; i++) {
      uint64_t cblk_size = std::min(npts - i * blk_size, blk_size);
      readr.read((char *) read_buf, cblk_size * ndims * sizeof(T));
#pragma omp parallel for schedule(static, 4096)
      for (uint64_t j = 0; j < cblk_size; j++) {
        normalize_data(read_buf + j * ndims, read_buf + j * ndims, ndims);
      }
      writr.write((char *) read_buf, cblk_size * ndims * sizeof(T));
    }
    delete[] read_buf;

    LOG(INFO) << "Wrote normalized points to file: " << outFileName;
  }

  /***************************************************
      Support for Merging Many Vamana Indices
   ***************************************************/

  void read_idmap(const std::string &fname, std::vector<unsigned> &ivecs) {
    uint32_t npts32, dim;
    size_t actual_file_size = get_file_size(fname);
    std::ifstream reader(fname.c_str(), std::ios::binary);
    reader.read((char *) &npts32, sizeof(uint32_t));
    reader.read((char *) &dim, sizeof(uint32_t));
    if (dim != 1 || actual_file_size != ((size_t) npts32) * sizeof(uint32_t) + 2 * sizeof(uint32_t)) {
      LOG(ERROR) << "Error reading idmap file. Check if the file is bin file with 1 dimensional data. Actual: "
                 << actual_file_size << ", expected: " << (size_t) npts32 + 2 * sizeof(uint32_t);

      crash();
    }
    ivecs.resize(npts32);
    reader.read((char *) ivecs.data(), ((size_t) npts32) * sizeof(uint32_t));
    reader.close();
  }

  int merge_shards(const std::string &vamana_prefix, const std::string &vamana_suffix, const std::string &idmaps_prefix,
                   const std::string &idmaps_suffix, const uint64_t nshards, unsigned max_degree,
                   const std::string &output_vamana, const std::string &medoids_file) {
    // Read ID maps
    std::vector<std::string> vamana_names(nshards);
    std::vector<std::vector<unsigned>> idmaps(nshards);
    for (uint64_t shard = 0; shard < nshards; shard++) {
      vamana_names[shard] = vamana_prefix + std::to_string(shard) + vamana_suffix;
      read_idmap(idmaps_prefix + std::to_string(shard) + idmaps_suffix, idmaps[shard]);
    }

    // find max node id
    uint64_t nnodes = 0;
    uint64_t nelems = 0;
    for (auto &idmap : idmaps) {
      for (auto &id : idmap) {
        nnodes = std::max(nnodes, (uint64_t) id);
      }
      nelems += idmap.size();
    }
    nnodes++;
    LOG(INFO) << "# nodes: " << nnodes << ", max. degree: " << max_degree;

    // compute inverse map: node -> shards
    std::vector<std::pair<unsigned, unsigned>> node_shard;
    node_shard.reserve(nelems);
    for (uint64_t shard = 0; shard < nshards; shard++) {
      LOG(INFO) << "Creating inverse map -- shard #" << shard;
      for (uint64_t idx = 0; idx < idmaps[shard].size(); idx++) {
        uint64_t node_id = idmaps[shard][idx];
        node_shard.push_back(std::make_pair((uint32_t) node_id, (uint32_t) shard));
      }
    }
    std::sort(node_shard.begin(), node_shard.end(), [](const auto &left, const auto &right) {
      return left.first < right.first || (left.first == right.first && left.second < right.second);
    });
    LOG(INFO) << "Finished computing node -> shards map";

    // create cached vamana readers
    std::vector<cached_ifstream> vamana_readers(nshards);
    for (uint64_t i = 0; i < nshards; i++) {
      vamana_readers[i].open(vamana_names[i], 1024 * 1048576);
      size_t expected_file_size;
      vamana_readers[i].read((char *) &expected_file_size, sizeof(uint64_t));
    }

    size_t merged_index_size = 24;
    size_t merged_index_frozen = 0;
    // create cached vamana writers
    cached_ofstream diskann_writer(output_vamana, 1024 * 1048576);
    diskann_writer.write((char *) &merged_index_size, sizeof(uint64_t));

    unsigned output_width = max_degree;
    unsigned max_input_width = 0;
    // read width from each vamana to advance buffer by sizeof(unsigned) bytes
    for (auto &reader : vamana_readers) {
      unsigned input_width;
      reader.read((char *) &input_width, sizeof(unsigned));
      max_input_width = input_width > max_input_width ? input_width : max_input_width;
    }

    LOG(INFO) << "Max input width: " << max_input_width << ", output width: " << output_width;

    diskann_writer.write((char *) &output_width, sizeof(unsigned));
    std::ofstream medoid_writer(medoids_file.c_str(), std::ios::binary);
    uint32_t nshards_u32 = (uint32_t) nshards;
    uint32_t one_val = 1;
    medoid_writer.write((char *) &nshards_u32, sizeof(uint32_t));
    medoid_writer.write((char *) &one_val, sizeof(uint32_t));

    uint64_t vamana_index_frozen = 0;
    for (uint64_t shard = 0; shard < nshards; shard++) {
      unsigned medoid;
      // read medoid
      vamana_readers[shard].read((char *) &medoid, sizeof(unsigned));
      vamana_readers[shard].read((char *) &vamana_index_frozen, sizeof(uint64_t));
      assert(vamana_index_frozen == false);
      // rename medoid
      medoid = idmaps[shard][medoid];

      medoid_writer.write((char *) &medoid, sizeof(uint32_t));
      // write renamed medoid
      if (shard == (nshards - 1))  //--> uncomment if running hierarchical
        diskann_writer.write((char *) &medoid, sizeof(unsigned));
    }
    diskann_writer.write((char *) &merged_index_frozen, sizeof(uint64_t));
    medoid_writer.close();

    LOG(INFO) << "Starting merge";

    // random_shuffle() is deprecated.
    std::random_device rng;
    std::mt19937 urng(rng());

    std::vector<bool> nhood_set(nnodes, 0);
    std::vector<unsigned> final_nhood;

    unsigned nnbrs = 0, shard_nnbrs = 0;
    unsigned cur_id = 0;
    for (const auto &id_shard : node_shard) {
      unsigned node_id = id_shard.first;
      unsigned shard_id = id_shard.second;
      if (cur_id < node_id) {
        // random_shuffle() is deprecated.
        std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
        nnbrs = (unsigned) std::min(final_nhood.size(), (uint64_t) max_degree);
        // write into merged ofstream
        diskann_writer.write((char *) &nnbrs, sizeof(unsigned));
        diskann_writer.write((char *) final_nhood.data(), nnbrs * sizeof(unsigned));
        merged_index_size += (sizeof(unsigned) + nnbrs * sizeof(unsigned));
        if (cur_id % 499999 == 1) {
          LOG(INFO) << cur_id << "...";
        }
        cur_id = node_id;
        nnbrs = 0;
        for (auto &p : final_nhood)
          nhood_set[p] = 0;
        final_nhood.clear();
      }
      // read from shard_id ifstream
      vamana_readers[shard_id].read((char *) &shard_nnbrs, sizeof(unsigned));
      std::vector<unsigned> shard_nhood(shard_nnbrs);
      vamana_readers[shard_id].read((char *) shard_nhood.data(), shard_nnbrs * sizeof(unsigned));

      // rename nodes
      for (uint64_t j = 0; j < shard_nnbrs; j++) {
        if (nhood_set[idmaps[shard_id][shard_nhood[j]]] == 0) {
          nhood_set[idmaps[shard_id][shard_nhood[j]]] = 1;
          final_nhood.emplace_back(idmaps[shard_id][shard_nhood[j]]);
        }
      }
    }

    // random_shuffle() is deprecated.
    std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
    nnbrs = (unsigned) std::min(final_nhood.size(), (uint64_t) max_degree);
    // write into merged ofstream
    diskann_writer.write((char *) &nnbrs, sizeof(unsigned));
    diskann_writer.write((char *) final_nhood.data(), nnbrs * sizeof(unsigned));
    merged_index_size += (sizeof(unsigned) + nnbrs * sizeof(unsigned));
    for (auto &p : final_nhood)
      nhood_set[p] = 0;
    final_nhood.clear();

    LOG(INFO) << "Expected size: " << merged_index_size;

    diskann_writer.reset();
    diskann_writer.write((char *) &merged_index_size, sizeof(uint64_t));

    LOG(INFO) << "Finished merge";
    return 0;
  }

  template<typename T>
  int build_merged_vamana_index(std::string base_file, pipeann::Metric _compareMetric, unsigned L, unsigned R,
                                double sampling_rate, double ram_budget, std::string mem_index_path,
                                std::string medoids_file, std::string centroids_file, const char *tag_file) {
    size_t base_num, base_dim;
    pipeann::get_bin_metadata(base_file, base_num, base_dim);

    double full_index_ram = estimate_ram_usage(base_num, base_dim, sizeof(T), R);
    if (full_index_ram < ram_budget * 1024 * 1024 * 1024) {
      LOG(INFO) << "Full index fits in RAM, building in one shot";
      pipeann::IndexBuildParameters paras;
      paras.set(R, L, 750, 1.2, 0, true);

      auto _pvamanaIndex = std::make_unique<pipeann::Index<T>>(_compareMetric, base_dim);
      // For SSD index, data is pre-normalized for correct PQ initialization, so normalize should be set to false.
      _pvamanaIndex->build(base_file.c_str(), base_num, paras, tag_file, false);
      _pvamanaIndex->save(mem_index_path.c_str());
      std::remove(medoids_file.c_str());
      std::remove(centroids_file.c_str());
      return 0;
    }

    std::string merged_index_prefix = mem_index_path + "_tempFiles";
    int num_parts =
        partition_with_ram_budget<T>(base_file, sampling_rate, ram_budget, 2 * R / 3, merged_index_prefix, 2);

    std::string cur_centroid_filepath = merged_index_prefix + "_centroids.bin";
    std::rename(cur_centroid_filepath.c_str(), centroids_file.c_str());

    for (int p = 0; p < num_parts; p++) {
      std::string shard_base_file = merged_index_prefix + "_subshard-" + std::to_string(p) + ".bin";
      std::string shard_index_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_mem.index";

      pipeann::IndexBuildParameters paras;
      paras.set(2 * R / 3, L, 750, 1.2, 0, false);
      uint64_t shard_base_dim, shard_base_pts;
      get_bin_metadata(shard_base_file, shard_base_pts, shard_base_dim);
      auto _pvamanaIndex = std::make_unique<pipeann::Index<T>>(_compareMetric, shard_base_dim);
      _pvamanaIndex->build(shard_base_file.c_str(), shard_base_pts, paras, nullptr, false);
      _pvamanaIndex->save(shard_index_file.c_str());
    }

    pipeann::merge_shards(merged_index_prefix + "_subshard-", "_mem.index", merged_index_prefix + "_subshard-",
                          "_ids_uint32.bin", num_parts, R, mem_index_path, medoids_file);

    // delete tempFiles
    for (int p = 0; p < num_parts; p++) {
      std::string shard_base_file = merged_index_prefix + "_subshard-" + std::to_string(p) + ".bin";
      std::string shard_id_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_ids_uint32.bin";
      std::string shard_index_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_mem.index";
      // Required if Index.cpp thinks we are building a multi-file index.
      std::string shard_index_file_data = shard_index_file + ".data";

      std::remove(shard_base_file.c_str());
      std::remove(shard_id_file.c_str());
      std::remove(shard_index_file.c_str());
      std::remove(shard_index_file_data.c_str());
    }
    return 0;
  }

  // if single_index format is true, we assume that the entire mem index is in
  // mem_index_file, and the entire disk index will be in output_file.
  template<typename T, typename TagT>
  void create_disk_layout(const std::string &mem_index_file, const std::string &base_file, const std::string &tag_file,
                          const std::string &output_file, AbstractLabel *label) {
    constexpr uint64_t kBlkSize = 64 * 1024 * 1024;

    // Open input files
    cached_ifstream base_reader;
    std::ifstream vamana_reader;
    base_reader.open(base_file, kBlkSize);
    vamana_reader.open(mem_index_file, std::ios::binary);

    // Read base file header
    uint32_t npts, ndims;
    base_reader.read((char *) &npts, sizeof(uint32_t));
    base_reader.read((char *) &ndims, sizeof(uint32_t));

    // Read vamana index header
    size_t index_file_size;
    uint32_t width, medoid;
    uint64_t vamana_frozen_num = 0;
    vamana_reader.read((char *) &index_file_size, sizeof(uint64_t));
    vamana_reader.read((char *) &width, sizeof(uint32_t));
    vamana_reader.read((char *) &medoid, sizeof(uint32_t));
    vamana_reader.read((char *) &vamana_frozen_num, sizeof(uint64_t));

    uint64_t label_size = (label != nullptr) ? label->label_size() : 0;

    // Compute disk layout parameters and build metadata
    uint64_t max_node_len = ((uint64_t) width + 1) * sizeof(uint32_t) + ndims * sizeof(T) + label_size;
    uint64_t nnodes_per_sector = SECTOR_LEN / max_node_len;  // 0 if max_node_len > SECTOR_LEN
    SSDIndexMetadata<T> meta(npts, ndims, medoid, max_node_len, nnodes_per_sector, label_size);
    meta.print();

    // Allocate sector buffer
    uint64_t bytes_per_write = nnodes_per_sector > 0 ? SECTOR_LEN : ROUND_UP(max_node_len, SECTOR_LEN);
    uint64_t nodes_per_write = nnodes_per_sector > 0 ? nnodes_per_sector : 1;
    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(bytes_per_write);

    // Create output file and write empty first sector (metadata placeholder)
    std::remove(output_file.c_str());
    cached_ofstream diskann_writer;
    diskann_writer.open(output_file, kBlkSize);
    memset(sector_buf.get(), 0, SECTOR_LEN);
    diskann_writer.write(sector_buf.get(), SECTOR_LEN);

    // Helper lambda to create DiskNode from buffer
    auto disk_node_at = [&](char *buf, uint32_t loc) -> DiskNode<T> { return DiskNode<T>(buf, loc, meta); };

    // Write all nodes
    for (uint64_t cur_node_id = 0; cur_node_id < meta.npoints;) {
      memset(sector_buf.get(), 0, bytes_per_write);

      uint64_t nodes_this_write = std::min(nodes_per_write, meta.npoints - cur_node_id);
      for (uint64_t i = 0; i < nodes_this_write; i++, cur_node_id++) {
        DiskNode<T> node = disk_node_at(sector_buf.get(), cur_node_id);

        // Read coords directly into DiskNode
        base_reader.read((char *) node.coords, meta.data_dim * sizeof(T));

        // Read nnbrs from vamana index
        uint32_t nnbrs_read;
        vamana_reader.read((char *) &nnbrs_read, sizeof(uint32_t));
        if (nnbrs_read == 0) {
          LOG(ERROR) << "Found point with no out-neighbors; Point#: " << cur_node_id;
          exit(-1);
        }

        // Read neighbors directly into DiskNode (truncate if exceeds width)
        uint32_t nnbrs_to_write = std::min(nnbrs_read, width);
        vamana_reader.read((char *) node.nbrs, nnbrs_to_write * sizeof(uint32_t));
        if (nnbrs_read > width) {
          vamana_reader.seekg((nnbrs_read - width) * sizeof(uint32_t), vamana_reader.cur);
        }
        node.nnbrs = nnbrs_to_write;

        if (label_size > 0) {
          label->write(static_cast<uint32_t>(cur_node_id), node.labels);
        }
      }

      diskann_writer.write(sector_buf.get(), bytes_per_write);
      if (cur_node_id % 100000 < nodes_per_write) {
        LOG(INFO) << "Nodes written: " << cur_node_id << "/" << meta.npoints;
      }
    }
    diskann_writer.close();

    // Handle tags file
    bool tags_enabled = !tag_file.empty();
    if (vamana_frozen_num > 0 || tags_enabled) {
      if (!file_exists(tag_file)) {
        LOG(ERROR) << "Tag file " << tag_file << " does not exist. Exiting...";
        exit(-1);
      }
      std::unique_ptr<TagT[]> mem_index_tags;
      size_t nr, nc;
      pipeann::load_bin<TagT>(tag_file, mem_index_tags, nr, nc, 0);
      if (nr != meta.npoints || nc != 1) {
        LOG(ERROR) << "Tag file dims mismatch: got " << nr << "x" << nc << ", expected " << meta.npoints << "x1";
        crash();
      }
      pipeann::save_bin<TagT>(output_file + ".tags", mem_index_tags.get(), nr, nc);
    }

    // Write metadata to the first sector
    meta.save_to_disk_index(output_file);
    LOG(INFO) << "Output file written.";
  }

  template<typename T, typename TagT>
  bool build_disk_index(const char *dataPath, const char *indexFilePath, uint32_t R, uint32_t L, uint32_t M,
                        uint32_t num_threads, uint32_t bytes_per_nbr, pipeann::Metric _compareMetric,
                        const char *tag_file, AbstractNeighbor<T> *nbr_handler, AbstractLabel *label) {
    std::string dataFilePath(dataPath);
    std::string index_prefix_path(indexFilePath);
    std::string mem_index_path = index_prefix_path + "_mem.index";
    std::string disk_index_path = index_prefix_path + "_disk.index";
    std::string medoids_path = disk_index_path + "_medoids.bin";
    std::string centroids_path = disk_index_path + "_centroids.bin";

    if (num_threads != 0) {
      omp_set_num_threads(num_threads);
    }

    LOG(INFO) << "Starting index build: R=" << R << " L=" << L << " Build RAM budget: " << M << "GB T: " << num_threads
              << " bytes per neighbor: " << bytes_per_nbr << " Final index will be in multiple files";

    std::string normalized_file_path = dataFilePath;
    // Normalize data for cosine metric.
    if (_compareMetric == pipeann::Metric::COSINE) {
      normalized_file_path = std::string(indexFilePath) + "_data.normalized.bin";
      normalize_data_file<T>(dataFilePath, normalized_file_path);
    }

    auto s = std::chrono::high_resolution_clock::now();
    nbr_handler->build(index_prefix_path, normalized_file_path, bytes_per_nbr);

    auto start = std::chrono::high_resolution_clock::now();
    auto p_val = nbr_handler->get_sample_p();
    pipeann::build_merged_vamana_index<T>(normalized_file_path, _compareMetric, L, R, p_val, M, mem_index_path,
                                          medoids_path, centroids_path, tag_file);
    auto end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Vamana index built in: " << std::chrono::duration<double>(end - start).count() << "s.";

    if (tag_file == nullptr) {
      pipeann::create_disk_layout<T, TagT>(mem_index_path, normalized_file_path, "", disk_index_path, label);
    } else {
      std::string tag_filename = std::string(tag_file);
      pipeann::create_disk_layout<T, TagT>(mem_index_path, normalized_file_path, tag_filename, disk_index_path, label);
    }

    LOG(INFO) << "Deleting memory index file: " << mem_index_path;
    std::remove(mem_index_path.c_str());
    // TODO: This is poor design. The decision to add the ".data" prefix
    // is taken by build_vamana_index. So, we shouldn't repeate it here.
    // Checking to see if we can merge the data and index into one file.
    std::remove((mem_index_path + ".data").c_str());
    if (normalized_file_path != dataFilePath) {
      // then we created a normalized vector file. Delete it.
      LOG(INFO) << "Deleting normalized vector file: " << normalized_file_path;
      std::remove(normalized_file_path.c_str());
    }

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    LOG(INFO) << "Indexing time: " << diff.count();
    return true;
  }

  template void create_disk_layout<int8_t, uint32_t>(const std::string &mem_index_file, const std::string &base_file,
                                                     const std::string &tag_file, const std::string &output_file,
                                                     AbstractLabel *label);
  template void create_disk_layout<uint8_t, uint32_t>(const std::string &mem_index_file, const std::string &base_file,
                                                      const std::string &tag_file, const std::string &output_file,
                                                      AbstractLabel *label);
  template void create_disk_layout<float, uint32_t>(const std::string &mem_index_file, const std::string &base_file,
                                                    const std::string &tag_file, const std::string &output_file,
                                                    AbstractLabel *label);

  template bool build_disk_index<int8_t, uint32_t>(const char *dataPath, const char *indexFilePath, uint32_t R,
                                                   uint32_t L, uint32_t M, uint32_t num_threads, uint32_t bytes_per_nbr,
                                                   pipeann::Metric _compareMetric, const char *tag_file,
                                                   AbstractNeighbor<int8_t> *nbr_handler, AbstractLabel *label);
  template bool build_disk_index<uint8_t, uint32_t>(const char *dataPath, const char *indexFilePath, uint32_t R,
                                                    uint32_t L, uint32_t M, uint32_t num_threads,
                                                    uint32_t bytes_per_nbr, pipeann::Metric _compareMetric,
                                                    const char *tag_file, AbstractNeighbor<uint8_t> *nbr_handler,
                                                    AbstractLabel *label);
  template bool build_disk_index<float, uint32_t>(const char *dataPath, const char *indexFilePath, uint32_t R,
                                                  uint32_t L, uint32_t M, uint32_t num_threads, uint32_t bytes_per_nbr,
                                                  pipeann::Metric _compareMetric, const char *tag_file,
                                                  AbstractNeighbor<float> *nbr_handler, AbstractLabel *label);

  template int build_merged_vamana_index<int8_t>(std::string base_file, pipeann::Metric _compareMetric, unsigned L,
                                                 unsigned R, double sampling_rate, double ram_budget,
                                                 std::string mem_index_path, std::string medoids_path,
                                                 std::string centroids_file, const char *tag_file);
  template int build_merged_vamana_index<float>(std::string base_file, pipeann::Metric _compareMetric, unsigned L,
                                                unsigned R, double sampling_rate, double ram_budget,
                                                std::string mem_index_path, std::string medoids_path,
                                                std::string centroids_file, const char *tag_file);
  template int build_merged_vamana_index<uint8_t>(std::string base_file, pipeann::Metric _compareMetric, unsigned L,
                                                  unsigned R, double sampling_rate, double ram_budget,
                                                  std::string mem_index_path, std::string medoids_path,
                                                  std::string centroids_file, const char *tag_file);
};  // namespace pipeann
