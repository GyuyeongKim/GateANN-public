#pragma once

#include "utils.h"
#include "utils/tsl/robin_set.h"

constexpr size_t MAX_N_SECTOR_READS = 128;
constexpr size_t MAX_N_EDGES = 1024;
constexpr size_t INDEX_SIZE_FACTOR = 2;  // space amplification during insert.

// Both unaligned and aligned.
// example: a record locates in [300, 500], then
// offset = 0, len = 4096 (aligned read for disk)
// u_offset = 300, u_len = 200 (unaligned read)
// Unaligned read: read u_len from u_offset, read to buf + 0.
struct IORequest {
  uint64_t offset;    // where to read from (page)
  uint64_t len;       // how much to read
  void *buf;          // where to read into
  bool finished;      // for async IO
  uint64_t u_offset;  // where to read from (unaligned)
  uint64_t u_len;     // how much to read (unaligned)
  void *base;         // starting address of this sector scratch

  IORequest() : offset(0), len(0), buf(nullptr) {
  }

  IORequest(uint64_t offset, uint64_t len, void *buf, uint64_t u_offset, uint64_t u_len, void *base = nullptr)
      : offset(offset), len(len), buf(buf), u_offset(u_offset), u_len(u_len), base(base) {
    assert((uint64_t) buf % SECTOR_LEN == 0);
    assert(offset % SECTOR_LEN == 0);
    assert(len % SECTOR_LEN == 0);
  }
};

namespace pipeann {
  template<typename T, typename IdT = uint32_t>
  struct SSDIndexMetadata {
    // The order matches that on SSD.
    uint32_t nr, nc;
    uint64_t npoints;  // size.
    uint64_t data_dim;
    uint64_t entry_point;
    uint64_t max_node_len;  // without data.
    uint64_t nnodes_per_sector;
    uint64_t npts_cur_shard;
    uint64_t label_size;
    uint64_t max_npts;  // capacity.
    uint64_t range;     // maximum out-degree.

    /* temporary fields (currently not stored on disk). */
    IdT entry_point_id;
    enum DataType : uint64_t { UNDEFINED = 0, FLOAT = 1, UINT8 = 2, INT8 = 3 } data_type;  // currently unused.

    SSDIndexMetadata() {
    }

    SSDIndexMetadata(uint64_t npoints, uint64_t data_dim, uint64_t entry_point, uint64_t max_node_len,
                     uint64_t nnodes_per_sector, uint64_t label_size = 0)
        : npoints(npoints), data_dim(data_dim), entry_point(entry_point), max_node_len(max_node_len),
          nnodes_per_sector(nnodes_per_sector), npts_cur_shard(npoints), label_size(label_size), data_type(UNDEFINED) {
      this->init_temporary_fields();
    }

    void init_temporary_fields() {
      this->max_npts = npoints;
      this->range = (max_node_len - data_dim * sizeof(T) - label_size) / sizeof(unsigned) - 1;
      this->entry_point_id = static_cast<IdT>(entry_point);
      assert(entry_point_id == entry_point);
    }

    void print() const {
      LOG(INFO) << "Max npts: " << max_npts << " Npoints: " << npoints << " Entry point: " << entry_point
                << " Data dim: " << data_dim << " Range: " << range;
      LOG(INFO) << "Max node len: " << max_node_len << " Nnodes per sector: " << nnodes_per_sector
                << " Npts cur shard: " << npts_cur_shard << " Label size: " << label_size;
    }

    void load_from_disk_index(const std::string &filename, bool sharded = false) {
      if (file_exists(filename) == false) {
        LOG(ERROR) << "File " << filename << " does not exist.";
        exit(-1);
      }
      std::ifstream in(filename, std::ios::binary);
      load_from_disk_index(in, sharded);
      in.close();
    }

    void load_from_disk_index(std::ifstream &in, bool sharded = false) {
      LOG(INFO) << "Loading metadata from disk index, sharded: " << sharded;
      in.read((char *) &nr, sizeof(uint32_t));
      in.read((char *) &nc, sizeof(uint32_t));

      in.read((char *) &npoints, sizeof(uint64_t));
      in.read((char *) &data_dim, sizeof(uint64_t));

      in.read((char *) &entry_point, sizeof(uint64_t));
      in.read((char *) &max_node_len, sizeof(uint64_t));
      in.read((char *) &nnodes_per_sector, sizeof(uint64_t));
      in.read((char *) &npts_cur_shard, sizeof(uint64_t));
      in.read((char *) &label_size, sizeof(uint64_t));

      if (!sharded) {
        this->npts_cur_shard = this->npoints;
      }

      if (nr < 7) {  // backward compatible.
        this->label_size = 0;
      }

      this->init_temporary_fields();
    }

    void save_to_disk_index(const std::string &filename) {
      std::ofstream out(filename, std::ios::in | std::ios::out | std::ios::binary);
      save_to_disk_index(out);
      out.close();
    }

    void save_to_disk_index(std::ofstream &out) {
      nr = 7;  // hard-coded for the number of uint64_t below.
      nc = 1;
      out.write((char *) &nr, sizeof(uint32_t));
      out.write((char *) &nc, sizeof(uint32_t));

      out.write((char *) &npoints, sizeof(uint64_t));
      out.write((char *) &data_dim, sizeof(uint64_t));

      out.write((char *) &entry_point, sizeof(uint64_t));
      out.write((char *) &max_node_len, sizeof(uint64_t));
      out.write((char *) &nnodes_per_sector, sizeof(uint64_t));
      out.write((char *) &npts_cur_shard, sizeof(uint64_t));
      out.write((char *) &label_size, sizeof(uint64_t));
    }
  };

  // The index is stored as fixed-size DiskNodes (records) on disk.
  // Each DiskNode contains: [vector (coords) | nnbrs | nnbrs neighbor IDs | labels (maybe 0 length) ].
  // This struct serves as a reference to a DiskNode<T> in the in-memory page-aligned buffer.
  template<typename T>
  struct DiskNode {
    T *coords;
    uint32_t &nnbrs;
    uint32_t *nbrs;
    void *labels;

    DiskNode<T>(char *page_buf, uint32_t loc, const SSDIndexMetadata<T> &meta)
        : coords((T *) (page_buf +
                        (meta.nnodes_per_sector == 0 ? 0 : (loc % meta.nnodes_per_sector) * meta.max_node_len))),
          nnbrs(*(uint32_t *) ((char *) coords + meta.data_dim * sizeof(T))),
          nbrs((uint32_t *) ((char *) coords + meta.data_dim * sizeof(T) + sizeof(uint32_t))),
          labels((void *) ((char *) coords + meta.data_dim * sizeof(T) + (1 + meta.range) * sizeof(uint32_t))) {
    }
  };

  template<typename T>
  struct QueryBuffer {
    T *coord_scratch = nullptr;  // MUST BE AT LEAST [aligned_dim], for current vector in comparison.

    char *sector_scratch = nullptr;  // MUST BE AT LEAST [MAX_N_SECTOR_READS * SECTOR_LEN], for sectors.
    uint64_t sector_idx = 0;         // index of next [SECTOR_LEN] scratch to use

    float *nbr_ctx_scratch = nullptr;       // MUST BE AT LEAST [256 * NCHUNKS], for pq table distance.
    float *aligned_dist_scratch = nullptr;  // MUST BE AT LEAST pipeann MAX_DEGREE, for exact dist.
    uint8_t *nbr_vec_scratch = nullptr;     // MUST BE AT LEAST  [N_CHUNKS * MAX_DEGREE], for neighbor PQ vectors.
    T *aligned_query_T = nullptr;
    char *update_buf = nullptr;  // Dynamic allocate in insert_in_place.

    tsl::robin_set<uint64_t> *visited = nullptr;
    tsl::robin_set<unsigned> *page_visited = nullptr;
    IORequest reqs[MAX_N_SECTOR_READS];

    void reset() {
      sector_idx = 0;
      visited->clear();  // does not deallocate memory.
      page_visited->clear();
    }
  };
};  // namespace pipeann
