#ifndef SELECTOR_H_
#define SELECTOR_H_

#include "ssd_index_defs.h"

namespace pipeann {
  /*
   * Selector is defined similar to Selector in faiss.
   * It is used to filter the results of the search (using query Label and target Label).
   */

  struct AbstractSelector {
    virtual ~AbstractSelector() = default;

    // Check if the target metadata meets the filter criteria.
    virtual bool is_member(uint32_t target_id, const void *query_labels, const void *target_labels) = 0;
  };

  // A dummy selector that always returns true.
  struct DummySelector : public AbstractSelector {
    bool is_member(uint32_t target_id, const void *query_labels, const void *target_labels) override {
      return true;
    }
  };

  // A simple range filter selector.
  // Query metadata is a range [low, high]; target metadata is a single value or nullptr.
  // The selector checks if the target metadata is within the range [low, high].
  // If target_labels is nullptr, returns false (no extra data).
  struct RangeSelector : public AbstractSelector {
    bool is_member(uint32_t target_id, const void *query_labels, const void *target_labels) override {
      if (unlikely(target_labels == nullptr)) {
        return false; /* nullptr means no extra data */
      }

      uint32_t low, high, target;
      memcpy(&low, (char *) query_labels, sizeof(uint32_t));
      memcpy(&high, (char *) query_labels + sizeof(uint32_t), sizeof(uint32_t));
      memcpy(&target, (char *) target_labels, sizeof(uint32_t));
      return target >= low && target <= high;
    }
  };

  // The selector checks if query and target label sets have non-empty intersection.
  // Assumptions:
  // - Query metadata: Contains label set Fq in format [count: uint32_t][label1: uint32_t]...[labelN: uint32_t]
  // - Target metadata: Contains label set Fx in format [count: uint32_t][label1: uint32_t]...[labelN: uint32_t]
  //   Labels may not be sorted and may contain duplicates
  struct LabelIntersectionSelector : public AbstractSelector {
    bool is_member(uint32_t target_id, const void *query_labels, const void *target_labels) override {
      uint32_t query_count, target_count;
      memcpy(&query_count, query_labels, sizeof(uint32_t));
      memcpy(&target_count, target_labels, sizeof(uint32_t));

      if (query_count == 0 || target_count == 0) {
        return false;
      }

      std::vector<uint32_t> query_labels_vec(query_count);
      std::vector<uint32_t> target_labels_vec(target_count);
      memcpy(query_labels_vec.data(), (char *) query_labels + sizeof(uint32_t), query_count * sizeof(uint32_t));
      memcpy(target_labels_vec.data(), (char *) target_labels + sizeof(uint32_t), target_count * sizeof(uint32_t));

      for (uint32_t q_idx = 0; q_idx < query_count; ++q_idx) {
        for (uint32_t t_idx = 0; t_idx < target_count; ++t_idx) {
          if (query_labels_vec[q_idx] == target_labels_vec[t_idx]) {
            return true;
          }
        }
      }
      return false;
    }
  };

  // The selector checks if query set is a subset of the target label set.
  // Could be used in NIPS 2023 bigann benchmark.
  // - Query metadata: Contains label set Fq in format [count: uint32_t][label1: uint32_t]...[labelN: uint32_t]
  // - Target metadata: Contains label set Fx in format [count: uint32_t][label1: uint32_t]...[labelN: uint32_t]
  struct LabelSubsetSelector : public AbstractSelector {
    bool is_member(uint32_t target_id, const void *query_labels, const void *target_labels) override {
      uint32_t query_count, target_count;

      memcpy(&query_count, query_labels, sizeof(uint32_t));
      memcpy(&target_count, target_labels, sizeof(uint32_t));

      if (query_count == 0) {
        return true;
      }
      if (target_count == 0) {
        return false;
      }

      std::vector<uint32_t> query_labels_vec(query_count);
      std::vector<uint32_t> target_labels_vec(target_count);
      memcpy(query_labels_vec.data(), (char *) query_labels + sizeof(uint32_t), query_count * sizeof(uint32_t));
      memcpy(target_labels_vec.data(), (char *) target_labels + sizeof(uint32_t), target_count * sizeof(uint32_t));

      for (uint32_t q_idx = 0; q_idx < query_count; ++q_idx) {
        bool found = false;
        for (uint32_t t_idx = 0; t_idx < target_count; ++t_idx) {
          if (query_labels_vec[q_idx] == target_labels_vec[t_idx]) {
            found = true;
            break;
          }
        }
        if (!found) {
          return false;
        }
      }
      return true;
    }
  };

  template<typename T>
  inline AbstractSelector *get_selector(const std::string &selector_type) {
    if (selector_type == "range") {
      return new RangeSelector();
    } else if (selector_type == "intersect") {
      return new LabelIntersectionSelector();
    } else if (selector_type == "subset") {
      return new LabelSubsetSelector();
    }
    return nullptr;
  }
}  // namespace pipeann

#endif