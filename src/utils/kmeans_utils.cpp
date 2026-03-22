#include <limits>
#include <malloc.h>
#include <cstring>
#include <utils/kmeans_utils.h>
#include <queue>
#include "distance.h"

namespace kmeans {
  struct PivotContainer {
    PivotContainer() = default;

    PivotContainer(size_t pivo_id, float pivo_dist) : piv_id{pivo_id}, piv_dist{pivo_dist} {
    }

    bool operator<(const PivotContainer &p) const {
      return p.piv_dist < piv_dist;
    }

    bool operator>(const PivotContainer &p) const {
      return p.piv_dist > piv_dist;
    }

    size_t piv_id;
    float piv_dist;
  };

  // compute l2-squared norms of data stored in row major num_points * dim,
  // needs to be pre-allocated
  void compute_vecs_l2sq(float *vecs_l2sq, float *data, const size_t num_points, const size_t dim) {
#pragma omp parallel for schedule(static, 8192)
    for (int64_t n_iter = 0; n_iter < (int64_t) num_points; n_iter++) {
      vecs_l2sq[n_iter] = cblas_sdot(dim, data + (n_iter * dim), 1, data + (n_iter * dim), 1);
    }
  }

  // Given data in num_points * dim row major and pivots in num_centers * dim row major,
  // calculate the k closest centers for each point and store in closest_centers_ivf
  // (row major, num_points * k). Uses efficient block-based parallel computation with BLAS.
  void compute_closest_centers(float *data, size_t num_points, size_t dim, float *pivot_data, size_t num_centers,
                               size_t k, uint32_t *closest_centers_ivf) {
    if (k > num_centers) {
      LOG(INFO) << "ERROR: k (" << k << ") > num_center(" << num_centers << ")";
      return;
    }

    // Compute L2 squared norms
    std::vector<float> pts_norms_squared(num_points);
    std::vector<float> pivs_norms_squared(num_centers);
    compute_vecs_l2sq(pts_norms_squared.data(), data, num_points, dim);
    compute_vecs_l2sq(pivs_norms_squared.data(), pivot_data, num_centers, dim);

    // Use a reasonable block size for good cache locality
    size_t PAR_BLOCK_SIZE = std::min((size_t) 8192, num_points);
    size_t N_BLOCKS = (num_points + PAR_BLOCK_SIZE - 1) / PAR_BLOCK_SIZE;

    // Read-only ones vector for BLAS broadcasting (shared by all threads)
    size_t ones_size = std::max(num_centers, PAR_BLOCK_SIZE);
    std::vector<float> ones(ones_size, 1.0f);

    // Parallelize over blocks
#pragma omp parallel
    {
      // Thread-local buffer (distance_matrix is read-write, must be per-thread)
      std::vector<float> distance_matrix(num_centers * PAR_BLOCK_SIZE);

#pragma omp for schedule(dynamic, 1)
      for (size_t cur_blk = 0; cur_blk < N_BLOCKS; cur_blk++) {
        size_t block_start = cur_blk * PAR_BLOCK_SIZE;
        size_t num_pts_blk = std::min(PAR_BLOCK_SIZE, num_points - block_start);
        const float *block_data = data + block_start * dim;
        const float *block_norms = pts_norms_squared.data() + block_start;

        // Compute squared distances using: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x^T*y

        // Step 1: dist_matrix = pts_norms * ones^T  (broadcast point norms)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, num_pts_blk, num_centers, 1, 1.0f, block_norms, 1,
                    ones.data(), 1, 0.0f, distance_matrix.data(), num_centers);

        // Step 2: dist_matrix += ones * pivs_norms^T  (broadcast pivot norms)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, num_pts_blk, num_centers, 1, 1.0f, ones.data(), 1,
                    pivs_norms_squared.data(), 1, 1.0f, distance_matrix.data(), num_centers);

        // Step 3: dist_matrix -= 2 * data * pivots^T  (inner products)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, num_pts_blk, num_centers, dim, -2.0f, block_data, dim,
                    pivot_data, dim, 1.0f, distance_matrix.data(), num_centers);

        // Find k nearest centers for each point and write directly to output
        if (k == 1) {
          // Optimized path for k=1 (most common case) - no intermediate buffer needed
          for (size_t i = 0; i < num_pts_blk; i++) {
            float min_dist = std::numeric_limits<float>::max();
            uint32_t min_idx = 0;
            const float *dists = distance_matrix.data() + i * num_centers;
            for (size_t j = 0; j < num_centers; j++) {
              if (dists[j] < min_dist) {
                min_idx = (uint32_t) j;
                min_dist = dists[j];
              }
            }
            closest_centers_ivf[block_start + i] = min_idx;
          }
        } else {
          // General case: find top-k using priority queue, write directly to output
          for (size_t i = 0; i < num_pts_blk; i++) {
            std::priority_queue<PivotContainer> top_k_queue;
            const float *dists = distance_matrix.data() + i * num_centers;
            for (size_t j = 0; j < num_centers; j++) {
              top_k_queue.push(PivotContainer(j, dists[j]));
            }
            uint32_t *output = closest_centers_ivf + (block_start + i) * k;
            for (size_t j = 0; j < k; j++) {
              output[j] = (uint32_t) top_k_queue.top().piv_id;
              top_k_queue.pop();
            }
          }
        }
      }
    }
  }
  // run Elkan one iteration
  // Elkan's algorithm uses triangle inequality to avoid unnecessary distance computations
  // Given data in row major num_points * dim, and centers in row major num_centers * dim
  // Uses lowerBound (num_points * num_centers), upperBound (num_points),
  // s (num_centers), halfcdist (num_centers * num_centers), newcdist (num_centers)
  // r: (num_points), if r[j] is true, it indicates that upper_bound[j] is loose
  //    and might need to be recomputed for tighter pruning.
  float elkan_iter(float *data, size_t num_points, size_t dim, float *centers, size_t num_centers,
                   std::vector<size_t> *closest_docs, uint32_t *&closest_center, float *lower_bound, float *upper_bound,
                   float *s_dist, float *half_center_dist, float *new_center_dist, float *new_centers, bool *r) {
    // closest_docs and closest_center are assumed to be non-null and allocated by the caller.
    pipeann::DistanceL2Float dist_cmp;

    // Clear the document lists for the new iteration
    for (size_t c = 0; c < num_centers; ++c) {
      closest_docs[c].clear();
    }

    // Step 1: Compute half the L2 distance between all pairs of centers.
#pragma omp parallel for schedule(static, 1)
    for (int64_t j = 0; j < static_cast<int64_t>(num_centers); j++) {
      s_dist[j] = std::numeric_limits<float>::max();  // Initialize s_dist for finding min
      for (size_t k = j + 1; k < num_centers; k++) {
        float distance = sqrtf(dist_cmp.compare(centers + j * dim, centers + k * dim, dim));
        half_center_dist[j * num_centers + k] = 0.5f * distance;
        half_center_dist[k * num_centers + j] = 0.5f * distance;
      }
    }

    // For all centers c, compute s(c) = min_{c' != c} d(c, c') / 2
#pragma omp parallel for schedule(static, 1)
    for (size_t j = 0; j < num_centers; j++) {
      float min_half_dist = std::numeric_limits<float>::max();
      for (size_t k = 0; k < num_centers; k++) {
        if (j == k)
          continue;
        if (half_center_dist[j * num_centers + k] < min_half_dist) {
          min_half_dist = half_center_dist[j * num_centers + k];
        }
      }
      s_dist[j] = min_half_dist;
    }

    // Step 2 & 3: For all points x, prune and re-assign
    int changes = 0;
#pragma omp parallel for schedule(static, 4096) reduction(+ : changes)
    for (int64_t j = 0; j < static_cast<int64_t>(num_points); j++) {
      uint32_t current_center_idx = closest_center[j];

      // First pruning: if d(x, c(x)) <= s(c(x)), no need to check other centers.
      if (upper_bound[j] <= s_dist[current_center_idx]) {
        continue;
      }

      float upper_bound_sq;
      // If r[j] is true, the upper bound is loose. We must recompute it to get a tighter bound.
      if (r[j]) {
        upper_bound_sq = dist_cmp.compare(data + j * dim, centers + current_center_idx * dim, dim);
        upper_bound[j] = sqrtf(upper_bound_sq);
        r[j] = false;  // The bound is now tight. Reset the flag.
        // Re-check the first pruning condition with the new tight bound.
        if (upper_bound[j] <= s_dist[current_center_idx]) {
          continue;
        }
      } else {
        upper_bound_sq = upper_bound[j] * upper_bound[j];
      }

      for (size_t k = 0; k < num_centers; k++) {
        if (k == current_center_idx)
          continue;

        // Second pruning: if d(x, c(x)) <= l(x, c'), skip distance calculation.
        if (upper_bound[j] <= lower_bound[j * num_centers + k]) {
          continue;
        }

        // Third pruning: if d(x, c(x)) <= 0.5 * d(c(x), c'), skip.
        if (upper_bound[j] <= half_center_dist[current_center_idx * num_centers + k]) {
          continue;
        }

        // If pruning fails, we must compute the exact distance.
        float dist_to_k_sq = dist_cmp.compare(data + j * dim, centers + k * dim, dim);
        lower_bound[j * num_centers + k] = sqrtf(dist_to_k_sq);

        if (dist_to_k_sq < upper_bound_sq) {
          current_center_idx = static_cast<uint32_t>(k);
          upper_bound_sq = dist_to_k_sq;
          upper_bound[j] = lower_bound[j * num_centers + k];
        }
      }

      if (current_center_idx != closest_center[j]) {
        closest_center[j] = current_center_idx;
        changes++;
      }
    }

    // Step 4: Update centers.
    // This part must be done after all points have been re-assigned.
    for (size_t j = 0; j < num_points; j++) {
      closest_docs[closest_center[j]].push_back(j);
    }

    // Use a temporary array for sum accumulation to allow parallel computation.
    std::vector<double> all_cluster_sums(num_centers * dim, 0.0);
#pragma omp parallel for schedule(static, 1)
    for (int64_t c = 0; c < static_cast<int64_t>(num_centers); ++c) {
      if (closest_docs[c].empty())
        continue;

      double *cluster_sum = all_cluster_sums.data() + c * dim;
      for (const auto &doc_id : closest_docs[c]) {
        float *current_doc = data + doc_id * dim;
        for (size_t d = 0; d < dim; d++) {
          cluster_sum[d] += static_cast<double>(current_doc[d]);
        }
      }
    }

#pragma omp parallel for schedule(static, 1)
    for (int64_t c = 0; c < static_cast<int64_t>(num_centers); ++c) {
      if (!closest_docs[c].empty()) {
        float *center = new_centers + c * dim;
        double *cluster_sum = all_cluster_sums.data() + c * dim;
        double cluster_size = static_cast<double>(closest_docs[c].size());
        for (size_t i = 0; i < dim; i++) {
          center[i] = static_cast<float>(cluster_sum[i] / cluster_size);
        }
      } else {
        // Handle empty clusters by keeping the old center.
        memcpy(new_centers + c * dim, centers + c * dim, dim * sizeof(float));
      }
    }

    // Step 5: Update lower bounds using the distance the centers moved.
#pragma omp parallel for schedule(static, 1)
    for (int64_t j = 0; j < static_cast<int64_t>(num_centers); j++) {
      new_center_dist[j] = sqrtf(dist_cmp.compare(centers + j * dim, new_centers + j * dim, dim));
    }

#pragma omp parallel for schedule(static, 8192)
    for (int64_t j = 0; j < static_cast<int64_t>(num_points); j++) {
      for (size_t k = 0; k < num_centers; k++) {
        float new_lower_bound = lower_bound[j * num_centers + k] - new_center_dist[k];
        lower_bound[j * num_centers + k] = (new_lower_bound > 0.0f) ? new_lower_bound : 0.0f;
      }
    }

    // Step 6: Update upper bounds and set r flags to true for next iteration.
#pragma omp parallel for schedule(static, 8192)
    for (int64_t j = 0; j < static_cast<int64_t>(num_points); j++) {
      upper_bound[j] += new_center_dist[closest_center[j]];
      r[j] = true;  // The upper bound is now loose, mark it for the next iteration.
    }

    // Step 7: Move new centers to the official centers array.
    memcpy(centers, new_centers, sizeof(float) * num_centers * dim);

    // Compute residual (sum of squared errors)
    float residual = 0.0;
    // This residual calculation logic from your code is correct and efficient.
    size_t BUF_PAD = 32;
    size_t CHUNK_SIZE = 2 * 8192;
    size_t nchunks = num_points / CHUNK_SIZE + (num_points % CHUNK_SIZE == 0 ? 0 : 1);
    std::vector<float> residuals(nchunks * BUF_PAD, 0.0);
#pragma omp parallel for schedule(static, 32)
    for (int64_t chunk = 0; chunk < static_cast<int64_t>(nchunks); ++chunk) {
      for (size_t d = chunk * CHUNK_SIZE; d < num_points && d < (chunk + 1) * CHUNK_SIZE; ++d) {
        residuals[chunk * BUF_PAD] +=
            dist_cmp.compare(data + (d * dim), centers + (size_t) closest_center[d] * (size_t) dim, dim);
      }
    }
    for (size_t chunk = 0; chunk < nchunks; ++chunk) {
      residual += residuals[chunk * BUF_PAD];
    }

    return residual;
  }

  // Run Elkan until max_reps or stopping criterion
  // Elkan's algorithm uses triangle inequality to avoid unnecessary distance computations
  // If you pass NULL for closest_docs and closest_center, it will NOT return the results,
  // else it will assume appropriate allocation as closest_docs = new vector<size_t> [num_centers],
  // and closest_center = new size_t[num_points]
  // Final centers are output in centers as row major num_centers * dim
  float run_elkan(float *data, size_t num_points, size_t dim, float *centers, const size_t num_centers,
                  const size_t max_reps, std::vector<size_t> *closest_docs, uint32_t *closest_center) {
    pipeann::DistanceL2Float dist_cmp;
    float residual = std::numeric_limits<float>::max();
    bool ret_closest_docs = (closest_docs != NULL);
    bool ret_closest_center = (closest_center != NULL);

    if (!ret_closest_docs) {
      closest_docs = new std::vector<size_t>[num_centers];
    }
    if (!ret_closest_center) {
      closest_center = new uint32_t[num_points];
    }

    // Allocate Elkan-specific arrays
    float *lower_bound = new float[num_points * num_centers];
    float *upper_bound = new float[num_points];
    float *s_dist = new float[num_centers];
    float *half_center_dist = new float[num_centers * num_centers];
    float *new_center_dist = new float[num_centers];
    float *new_centers = new float[num_centers * dim];
    bool *r = new bool[num_points];  // The crucial flag array for each point

    // Initialize lower_bound to zero
    memset(lower_bound, 0, sizeof(float) * num_points * num_centers);

    // Initial assignment: assign each point to its closest initial center
    // and initialize upper_bound with the exact L2 distance.
    for (int64_t j = 0; j < static_cast<int64_t>(num_points); j++) {
      float min_dist_sq = std::numeric_limits<float>::max();
      uint32_t best_center_idx = 0;

      for (size_t k = 0; k < num_centers; k++) {
        float dist_sq = dist_cmp.compare(data + j * dim, centers + k * dim, dim);
        if (dist_sq < min_dist_sq) {
          min_dist_sq = dist_sq;
          best_center_idx = static_cast<uint32_t>(k);
        }
      }
      // Store the TRUE L2 distance in upper_bound
      upper_bound[j] = sqrtf(min_dist_sq);
      closest_center[j] = best_center_idx;
    }

    // Initially, all upper bounds are tight, so no need to recompute. Set all r flags to false.
    memset(r, false, sizeof(bool) * num_points);

    float old_residual;
    for (size_t i = 0; i < max_reps; ++i) {
      old_residual = residual;

      residual = elkan_iter(data, num_points, dim, centers, num_centers, closest_docs, closest_center, lower_bound,
                            upper_bound, s_dist, half_center_dist, new_center_dist, new_centers, r);

      // Convergence check
      if (i > 0 && residual > std::numeric_limits<float>::epsilon()) {
        if (fabs((old_residual - residual) / residual) < 1e-5) {
          break;
        }
      } else if (residual < std::numeric_limits<float>::epsilon()) {
        break;
      }
    }

    // Cleanup
    delete[] lower_bound;
    delete[] upper_bound;
    delete[] s_dist;
    delete[] half_center_dist;
    delete[] new_center_dist;
    delete[] new_centers;
    delete[] r;
    if (!ret_closest_docs)
      delete[] closest_docs;
    if (!ret_closest_center)
      delete[] closest_center;

    return residual;
  }

  void kmeanspp_selecting_pivots(float *data, size_t num_points, size_t dim, float *pivot_data, size_t num_centers) {
    if (num_points > 1 << 23) {
      LOG(INFO) << "ERROR: n_pts " << num_points << " currently not supported for k-means++, maximum is 8388608.";
      return;
    }
    pipeann::DistanceL2Float dist_cmp;

    std::vector<size_t> picked;
    std::random_device rd;
    auto x = rd();
    std::mt19937 generator(x);
    std::uniform_real_distribution<> distribution(0, 1);
    std::uniform_int_distribution<size_t> int_dist(0, num_points - 1);
    size_t init_id = int_dist(generator);
    size_t num_picked = 1;

    picked.push_back(init_id);
    std::memcpy(pivot_data, data + init_id * dim, dim * sizeof(float));

    float *dist = new float[num_points];

    for (int64_t i = 0; i < (int64_t) num_points; i++) {
      dist[i] = dist_cmp.compare(data + i * dim, data + init_id * dim, dim);
    }

    double dart_val;
    size_t tmp_pivot;
    bool sum_flag = false;

    while (num_picked < num_centers) {
      dart_val = distribution(generator);

      double sum = 0;
      for (size_t i = 0; i < num_points; i++) {
        sum = sum + (double) dist[i];
      }
      if (sum == 0)
        sum_flag = true;

      dart_val *= sum;

      double prefix_sum = 0;
      for (size_t i = 0; i < (num_points); i++) {
        tmp_pivot = i;
        if (dart_val >= prefix_sum && dart_val < prefix_sum + (double) dist[i]) {
          break;
        }

        prefix_sum += (double) dist[i];
      }

      if (std::find(picked.begin(), picked.end(), tmp_pivot) != picked.end() && (sum_flag == false))
        continue;
      picked.push_back(tmp_pivot);
      std::memcpy(pivot_data + num_picked * dim, data + tmp_pivot * dim, dim * sizeof(float));

      for (int64_t i = 0; i < (int64_t) num_points; i++) {
        dist[i] = std::min(dist[i], dist_cmp.compare(data + i * dim, data + tmp_pivot * dim, dim));
      }
      num_picked++;
    }
    delete[] dist;
  }

}  // namespace kmeans
