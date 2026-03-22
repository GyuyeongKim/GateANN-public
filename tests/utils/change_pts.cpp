#include <omp.h>
#include <cstring>
#include <ctime>
#include "utils/timer.h"
#include "utils/log.h"
#include "utils.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

template<class T>
void convert(char *path, uint64_t to_pts) {
  int num_points, dim;
  std::ifstream reader(path, std::ios::binary | std::ios::ate);
  reader.seekg(0, std::ios::beg);
  reader.read((char *) &num_points, sizeof(int));
  reader.read((char *) &dim, sizeof(int));
  LOG(INFO) << "Num points: " << num_points << " Dim: " << dim << " To pts: " << to_pts;
  T *data = new T[to_pts * dim];
  reader.read((char *) data, to_pts * dim * sizeof(T));
  pipeann::save_bin<T>((std::string(path) + std::to_string(to_pts)).c_str(), data, to_pts, dim);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cout << "Correct usage: " << argv[0] << " <type[uint8/int8/float]> <file> <to_pts>" << std::endl;
    exit(-1);
  }

  int arg_no = 1;
  char *type = argv[arg_no++];
  char *base_data_file = argv[arg_no++];
  uint64_t to_pts = std::stoull(argv[arg_no++]);

  if (strcmp(type, "uint8") == 0) {
    convert<uint8_t>(base_data_file, to_pts);
  } else if (strcmp(type, "int8") == 0) {
    convert<int8_t>(base_data_file, to_pts);
  } else if (strcmp(type, "float") == 0) {
    convert<float>(base_data_file, to_pts);
  } else {
    std::cout << "Unknown type: " << type << std::endl;
    exit(-1);
  }
}