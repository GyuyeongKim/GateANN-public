#pragma once

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include "utils.h"

namespace pipeann {
  template<typename T>
  class Distance {
   public:
    virtual float compare(const T *a, const T *b, unsigned length) const = 0;
    virtual ~Distance() {
    }
  };

  class DistanceCosineInt8 : public Distance<int8_t> {
   public:
    virtual float compare(const int8_t *a, const int8_t *b, uint32_t length) const;
  };

  class DistanceCosineFloat : public Distance<float> {
   public:
    virtual float compare(const float *a, const float *b, uint32_t length) const;
  };

  class DistanceCosineUInt8 : public Distance<uint8_t> {
   public:
    virtual float compare(const uint8_t *a, const uint8_t *b, uint32_t length) const;
  };

  class DistanceL2Int8 : public Distance<int8_t> {
   public:
    virtual float compare(const int8_t *a, const int8_t *b, uint32_t size) const;
  };

  class DistanceL2UInt8 : public Distance<uint8_t> {
   public:
    virtual float compare(const uint8_t *a, const uint8_t *b, uint32_t size) const;
  };

  class DistanceL2Float : public Distance<float> {
   public:
    virtual float compare(const float *a, const float *b, uint32_t size) const __attribute__((hot));
  };

  template<typename T>
  class DistanceInnerProduct : public Distance<T> {
   public:
    virtual float compare(const T *a, const T *b, unsigned size);
  };

  inline Metric get_metric(const std::string &metric_str) {
    if (metric_str == "l2") {
      return Metric::L2;
    } else if (metric_str == "cosine") {
      return Metric::COSINE;
    } else if (metric_str == "mips") {
      return Metric::INNER_PRODUCT;
    } else {
      LOG(ERROR) << "Unsupported metric: " << metric_str << ". Using L2.";
      return Metric::L2;
    }
  }

  inline std::string get_metric_str(Metric m) {
    switch (m) {
      case Metric::L2:
        return "l2";
      case Metric::COSINE:
        return "cosine";
      case Metric::INNER_PRODUCT:
        return "mips";
      default:
        return "unknown";
    }
  }
  // The distance function does not return the actual distance, but reserves the partial order.
  // For L2, it returns the squared L2 distance.
  // For IP, it returns A^2 - inner_product, where A is 1 if float, 255 if uint8_t, and 127 if int8_t.
  // For cosine, it returns A^2 * (1 - cosine(theta)). Data should be first normalized by normalize_data.
  // Note that cosine distance function is used for both inner_product and cosine metrics.
  template<typename T>
  Distance<T> *get_distance_function(Metric m);
}  // namespace pipeann
