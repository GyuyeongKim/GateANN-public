#ifndef LABEL_WRITER_H_
#define LABEL_WRITER_H_

#include "ssd_index_defs.h"
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>

namespace pipeann {
  /*
   * Labels are used for filtered ANNS.
   * On SSD, it is stored after the neighbor IDs.
   * During a query, it is passed as a void * parameter for filtering.
   * Selector is used to select the labels.
   */

  // An abstract label interface.
  struct AbstractLabel {
    virtual ~AbstractLabel() = default;

    // Write labels to the provided buffer.
    // Use this for index building.
    virtual void write(uint32_t id, void *buffer) = 0;
    // Returns the maximum size of the buffer.
    virtual size_t label_size() = 0;
  };

  // A dummy label that does nothing.
  struct DummyLabel : public AbstractLabel {
    void write(uint32_t id, void *buffer) override {
      return;
    }

    size_t label_size() override {
      return 0;
    }
  };

  // A label that reads from spmat format.
  // spmat format: x[i][j] != 0 means vector i contains label j.
  struct SpmatLabel : public AbstractLabel {
    std::vector<std::vector<uint32_t>> labels_;  // labels_[i] contains all labels for vector i
    size_t max_label_count_ = 0;                 // maximum number of labels per vector
    size_t label_size_ = 0;                      // maximum buffer size needed

    SpmatLabel(const std::string &filename) {
      std::ifstream reader(filename, std::ios::binary);
      if (!reader.is_open()) {
        LOG(ERROR) << "Failed to open spmat file: " << filename;
        crash();
      }

      // Read header: 3 int64 values (nrow, ncol, nnz)
      int64_t nrow, ncol, nnz;
      reader.read((char *) &nrow, sizeof(int64_t));
      reader.read((char *) &ncol, sizeof(int64_t));
      reader.read((char *) &nnz, sizeof(int64_t));

      LOG(INFO) << "Loading spmat: nrow=" << nrow << ", ncol=" << ncol << ", nnz=" << nnz;

      // Read indptr: nrow+1 int64 values
      std::vector<int64_t> indptr(nrow + 1);
      reader.read((char *) indptr.data(), (nrow + 1) * sizeof(int64_t));

      // Read indices: nnz int32 values
      std::vector<int32_t> indices(nnz);
      reader.read((char *) indices.data(), nnz * sizeof(int32_t));

      // Read data: nnz float32 values
      std::vector<float> data(nnz);
      reader.read((char *) data.data(), nnz * sizeof(float));

      reader.close();

      // Process the sparse matrix: for each row i, collect labels where data != 0
      labels_.resize(nrow);
      max_label_count_ = 0;

      for (int64_t i = 0; i < nrow; i++) {
        int64_t start = indptr[i];
        int64_t end = indptr[i + 1];
        std::vector<uint32_t> row_labels;

        for (int64_t j = start; j < end; j++) {
          // x[i][j] != 0 means vector i contains label indices[j]
          if (data[j] != 0.0f) {
            row_labels.push_back(static_cast<uint32_t>(indices[j]));
          }
        }

        labels_[i] = std::move(row_labels);
        if (labels_[i].size() > max_label_count_) {
          max_label_count_ = labels_[i].size();
        }
      }

      // Calculate label_size: 4 bytes for label count + max_label_count * sizeof(uint32_t)
      label_size_ = sizeof(uint32_t) + max_label_count_ * sizeof(uint32_t);

      LOG(INFO) << "Loaded spmat labels: max_label_count=" << max_label_count_ << ", label_size=" << label_size_;
    }

    void write(uint32_t id, void *buffer) override {
      if (id >= labels_.size()) {
        LOG(ERROR) << "Label id " << id << " out of range (max: " << labels_.size() << ")";
        crash();
      }

      char *buf = (char *) buffer;
      const auto &label_list = labels_[id];

      // Write label count (4 bytes)
      uint32_t label_count = label_list.size();
      memcpy(buf, &label_count, sizeof(uint32_t));

      // Write labels
      if (label_count > 0) {
        memcpy(buf + sizeof(uint32_t), label_list.data(), label_count * sizeof(uint32_t));
      }
    }

    size_t label_size() override {
      return label_size_;
    }
  };

  inline pipeann::AbstractLabel *get_label(const std::string &label_type, const std::string &label_file) {
    if (label_type == "spmat") {
      return new pipeann::SpmatLabel(label_file);
    }
    return nullptr;
  }
}  // namespace pipeann

#endif  // LABEL_WRITER_H_