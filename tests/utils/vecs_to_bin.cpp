#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <string>
#include "utils.h"

// Generic block conversion using byte buffers. Each record layout:
// [4-byte unsigned dims][ndims * elem_size bytes]
static void block_convert(std::ifstream &reader, std::ofstream &writer, char *read_buf, char *write_buf, uint64_t npts,
                          uint64_t ndims, size_t elem_size) {
  const uint64_t rec_size_bytes = sizeof(unsigned) + ndims * elem_size;
  const uint64_t in_bytes = npts * rec_size_bytes;
  const uint64_t out_bytes = npts * ndims * elem_size;

  reader.read(read_buf, static_cast<std::streamsize>(in_bytes));
  for (uint64_t i = 0; i < npts; i++) {
    const char *src = read_buf + i * rec_size_bytes + sizeof(unsigned);
    char *dst = write_buf + i * (ndims * elem_size);
    std::memcpy(dst, src, ndims * elem_size);
  }
  writer.write(write_buf, static_cast<std::streamsize>(out_bytes));
}

static int convert_file(const std::string &in_path, const std::string &out_path, size_t elem_size) {
  std::ifstream reader(in_path, std::ios::binary | std::ios::ate);
  if (!reader) {
    std::cerr << "Failed to open input file: " << in_path << std::endl;
    return -1;
  }

  const uint64_t fsize = static_cast<uint64_t>(reader.tellg());
  reader.seekg(0, std::ios::beg);

  unsigned ndims_u32 = 0;
  reader.read(reinterpret_cast<char *>(&ndims_u32), sizeof(unsigned));
  if (!reader) {
    std::cerr << "Failed to read header from input file." << std::endl;
    return -1;
  }
  reader.seekg(0, std::ios::beg);
  const uint64_t ndims = static_cast<uint64_t>(ndims_u32);

  // Compute number of points based on record size = 4 + ndims * elem_size
  const uint64_t rec_bytes = sizeof(unsigned) + ndims * elem_size;
  if (rec_bytes == 0) {
    std::cerr << "Invalid record size." << std::endl;
    return -1;
  }
  const uint64_t npts = fsize / rec_bytes;

  std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims << std::endl;

  const uint64_t blk_size = 131072;  // number of records per block
  const uint64_t nblks = ROUND_UP(npts, blk_size) / blk_size;
  std::cout << "# blks: " << nblks << std::endl;

  std::ofstream writer(out_path, std::ios::binary);
  if (!writer) {
    std::cerr << "Failed to open output file: " << out_path << std::endl;
    return -1;
  }
  const int npts_s32 = static_cast<int32_t>(npts);
  const int ndims_s32 = static_cast<int32_t>(ndims);
  writer.write(reinterpret_cast<const char *>(&npts_s32), sizeof(int32_t));
  writer.write(reinterpret_cast<const char *>(&ndims_s32), sizeof(int32_t));

  // Allocate buffers for one block and reuse.
  const uint64_t in_blk_bytes = blk_size * (sizeof(unsigned) + ndims * elem_size);
  const uint64_t out_blk_bytes = blk_size * (ndims * elem_size);
  char *read_buf = new (std::nothrow) char[in_blk_bytes];
  char *write_buf = new (std::nothrow) char[out_blk_bytes];
  if (read_buf == nullptr || write_buf == nullptr) {
    std::cerr << "Failed to allocate buffers." << std::endl;
    delete[] read_buf;
    delete[] write_buf;
    return -1;
  }

  for (uint64_t i = 0; i < nblks; i++) {
    const uint64_t cblk_size = std::min(npts - i * blk_size, blk_size);
    block_convert(reader, writer, read_buf, write_buf, cblk_size, ndims, elem_size);
    std::cout << "Block #" << i << " written" << std::endl;
  }

  delete[] read_buf;
  delete[] write_buf;
  reader.close();
  writer.close();
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cout << argv[0] << " type(float|int32|uint32|int8|uint8) input_vecs output_bin" << std::endl;
    return -1;
  }

  const std::string type = argv[1];
  const std::string in_path = argv[2];
  const std::string out_path = argv[3];

  size_t elem_size = 0;
  if (type == "float" || type == "int32" || type == "uint32") {
    elem_size = 4;
  } else if (type == "int8" || type == "uint8") {
    elem_size = 1;
  } else {
    std::cerr << "Unknown type: " << type << std::endl;
    return -1;
  }

  return convert_file(in_path, out_path, elem_size);
}
