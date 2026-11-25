// F104 source-fix1 modification, 2026-10-09: bounded development derivative.
// sherpa-onnx/csrc/fast-clustering.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/fast-clustering.h"

#include <vector>

#include "Eigen/Dense"
#include "sherpa-onnx/csrc/f104-diarization-guard.h"
#include "fastcluster-all-in-one.h"  // NOLINT

namespace sherpa_onnx {

class FastClustering::Impl {
 public:
  explicit Impl(const FastClusteringConfig &config) : config_(config) {}

  std::vector<int32_t> Cluster(float *features, int32_t num_rows,
                               int32_t num_cols) const {
    f104::PositiveRows(num_rows, num_cols);
    f104::Require(features && config_.num_clusters == -1 && config_.threshold == 0.5f,
                  "UNREVIEWED_CLUSTERING_PARAMETERS");
    for (int32_t i = 0; i < num_rows; ++i) f104::Normalize(features + i*512, 512);
    if (num_rows == 1) return {0};

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        m(features, num_rows, num_cols);


    std::vector<double> distance(f104::DistanceCount(num_rows));

    uint64_t k = 0;
    for (int32_t i = 0; i != num_rows; ++i) {
      auto v = m.row(i);
      for (int32_t j = i + 1; j != num_rows; ++j) {
        double cosine_similarity = v.dot(m.row(j));
        double consine_dissimilarity = 1 - cosine_similarity;

        if (consine_dissimilarity < 0) {
          consine_dissimilarity = 0;
        }

        distance[k] = consine_dissimilarity;
        ++k;
      }
    }

    std::vector<int32_t> merge(2 * (num_rows - 1));
    std::vector<double> height(num_rows - 1);

    fastclustercpp::hclust_fast(num_rows, distance.data(),
                                fastclustercpp::HCLUST_METHOD_COMPLETE,
                                merge.data(), height.data());

    std::vector<int32_t> labels(num_rows);
    if (config_.num_clusters > 0) {
      fastclustercpp::cutree_k(num_rows, merge.data(), config_.num_clusters,
                               labels.data());
    } else {
      fastclustercpp::cutree_cdist(num_rows, merge.data(), height.data(),
                                   config_.threshold, labels.data());
    }

    f104::GuardClusterLabels(labels, num_rows);
    return labels;
  }

 private:
  FastClusteringConfig config_;
};

FastClustering::FastClustering(const FastClusteringConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

FastClustering::~FastClustering() = default;

std::vector<int32_t> FastClustering::Cluster(float *features, int32_t num_rows,
                                             int32_t num_cols) const {
  return impl_->Cluster(features, num_rows, num_cols);
}
}  // namespace sherpa_onnx
