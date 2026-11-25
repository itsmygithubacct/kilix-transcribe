// F104 source-fix1 modification, 2026-10-09: bounded development derivative.
// F104 source-fix2, 2026-10-09: validate score/activity/turns before success.
// sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_PYANNOTE_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_PYANNOTE_IMPL_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "sherpa-onnx/csrc/f104-diarization-guard.h"
#include "sherpa-onnx/csrc/f104-diagnostic-validation.h"
#include "sherpa-onnx/csrc/fast-clustering.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/offline-speaker-diarization-impl.h"
#include "sherpa-onnx/csrc/offline-speaker-segmentation-pyannote-model.h"
#include "sherpa-onnx/csrc/speaker-embedding-extractor.h"
#include "sherpa-onnx/csrc/timer.h"

namespace sherpa_onnx {

namespace {  // NOLINT

// copied from https://github.com/k2-fsa/k2/blob/master/k2/csrc/host/util.h#L41
template <class T>
inline void hash_combine(std::size_t *seed, const T &v) {  // NOLINT
  std::hash<T> hasher;
  *seed ^= hasher(v) + 0x9e3779b9 + ((*seed) << 6) + ((*seed) >> 2);  // NOLINT
}

// copied from https://github.com/k2-fsa/k2/blob/master/k2/csrc/host/util.h#L47
struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &pair) const {
    std::size_t result = 0;
    hash_combine(&result, pair.first);
    hash_combine(&result, pair.second);
    return result;
  }
};
}  // namespace

using Matrix2D = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>;  // NOLINT

using Matrix2DInt32 = Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>;  // NOLINT

using FloatRowVector = Eigen::Matrix<float, 1, Eigen::Dynamic>;
using Int32RowVector = Eigen::Matrix<int32_t, 1, Eigen::Dynamic>;

using Int32Pair = std::pair<int32_t, int32_t>;

class OfflineSpeakerDiarizationPyannoteImpl
    : public OfflineSpeakerDiarizationImpl {
 public:
  ~OfflineSpeakerDiarizationPyannoteImpl() override = default;

  explicit OfflineSpeakerDiarizationPyannoteImpl(
      const OfflineSpeakerDiarizationConfig &config)
      : config_(config),
        segmentation_model_(config_.segmentation),
        embedding_extractor_(config_.embedding),
        clustering_(std::make_unique<FastClustering>(config_.clustering)) {
    Init();
  }

  template <typename Manager>
  OfflineSpeakerDiarizationPyannoteImpl(
      Manager *mgr, const OfflineSpeakerDiarizationConfig &config)
      : config_(config),
        segmentation_model_(mgr, config_.segmentation),
        embedding_extractor_(mgr, config_.embedding),
        clustering_(std::make_unique<FastClustering>(config_.clustering)) {
    Init();
  }

  int32_t SampleRate() const override {
    const auto &meta_data = segmentation_model_.GetModelMetaData();

    return meta_data.sample_rate;
  }

  void SetConfig(const OfflineSpeakerDiarizationConfig &) override {
    throw std::runtime_error("IMMUTABLE_ALGORITHM_PARAMETERS: new reviewed tuple required");
  }

  OfflineSpeakerDiarizationResult Process(
      const float *audio, int32_t n,
      OfflineSpeakerDiarizationProgressCallback callback = nullptr,
      void *callback_arg = nullptr) const override {
    f104::Diagnostics d;
    auto result = ProcessDetailed(audio, n, &d, callback, callback_arg);
    // Legacy result cannot express unknown identity: never silently erase it.
    f104::Require(d.status == "OK" || d.status == "SILENCE", "DIAGNOSTICS_REQUIRED: unidentified activity");
    return result;
  }

  OfflineSpeakerDiarizationResult ProcessDetailed(
      const float *audio, int32_t n, f104::Diagnostics *d,
      OfflineSpeakerDiarizationProgressCallback callback = nullptr,
      void *callback_arg = nullptr) const {
    f104::Require(d != nullptr, "MISSING_DIAGNOSTIC_SINK");
    *d = f104::Diagnostics{}; d->samples = n;
    f104::ValidatePcm(audio, n);
    const auto &meta = segmentation_model_.GetModelMetaData();
    f104::Envelope(n, meta.window_size, meta.window_shift, 589, meta.num_classes,
                   meta.num_speakers, meta.receptive_field_shift);
    auto segmentations = RunSpeakerSegmentationModel(audio, n);
    std::vector<Matrix2DInt32> labels; labels.reserve(segmentations.size());
    for (int32_t i = 0; i < static_cast<int32_t>(segmentations.size()); ++i) {
      const auto &m = segmentations[i];
      auto bytes = f104::MatrixBytes(m.rows(), m.cols());
      d->score_bytes = f104::Add(d->score_bytes, bytes, f104::kDiagnosticBytes);
      f104::ChunkScores score;score.chunk=i;score.offset=i*meta.window_shift;
      score.frames=m.rows();score.classes=m.cols();
      score.values.assign(m.data(), m.data()+m.size());
      d->scores.push_back(std::move(score));
      auto label = ToMultiLabel(m);
      int64_t local_n = std::min<int64_t>(meta.window_size, n-i*meta.window_shift);
      int64_t keep = f104::KeepFrames(label.rows(), local_n, meta.receptive_field_size, meta.receptive_field_shift);
      if (keep < label.rows()) label.bottomRows(label.rows()-keep).setZero();
      labels.push_back(std::move(label));
    }
    segmentations.clear(); segmentations.shrink_to_fit();
    CaptureActivity(labels, n, d);  // includes original two-speaker overlap
    if (d->activity.empty()) {d->status="SILENCE";f104::ValidateDiagnostics(*d);return {};}
    // One chunk uses the same embedding/clustering/diagnostic path as all others.
    auto samples = GetChunkSpeakerSampleIndexes(labels, n);
    std::vector<int32_t> valid;
    Matrix2D embeddings = ComputeEmbeddings(audio, n, samples.first, samples.second, &valid, d, callback, callback_arg);
    if (embeddings.rows() == 0) {
      d->status="UNIDENTIFIABLE_ACTIVITY";
      f104::ValidateDiagnostics(*d);
      return {};  // raw evidence and explicit unknown activity remain in d
    }
    f104::PositiveRows(embeddings.rows(), embeddings.cols());
    std::vector<Int32Pair> pairs; pairs.reserve(valid.size());
    for (auto index : valid) {
      f104::Require(index>=0 && index<static_cast<int32_t>(samples.first.size()), "MALFORMED_VALID_INDEX");
      pairs.push_back(samples.first[index]);
    }
    auto clusters = clustering_->Cluster(embeddings.data(), embeddings.rows(), embeddings.cols());
    f104::GuardClusterLabels(clusters, embeddings.rows());
    auto mapping = ConvertChunkSpeakerToCluster(pairs, clusters);
    d->assignments = f104::AssignmentEvidence(embeddings.data(), embeddings.rows(), clusters, pairs);
    bool unknown = false;
    for (auto &m : d->mappings) {
      auto it = mapping.find({m.chunk,m.slot});
      if (it==mapping.end()) {unknown=true;continue;}
      m.cluster=it->second;m.reason="MAPPED";
    }
    for (auto &a : d->activity) {
      auto it=mapping.find({a.chunk,a.slot});if(it!=mapping.end())a.cluster=it->second;
    }
    auto relabeled = ReLabel(labels, *std::max_element(clusters.begin(),clusters.end()), mapping);
    auto count = ComputeSpeakerCount(relabeled,n);
    auto per_frame = ComputeSpeakersPerFrame(labels);
    auto final_labels = FinalizeLabels(count, per_frame);
    auto result = ComputeResult(final_labels,n,d);
    d->status=unknown ? "PARTIAL_UNKNOWN" : "OK";
    f104::ValidateDiagnostics(*d);
    return result;
  }


 private:
  void LogTotal(float total_seconds, int32_t num_samples) const {
    float audio_duration = static_cast<float>(num_samples) / SampleRate();

#if __OHOS__
    SHERPA_ONNX_LOGE(
        "OfflineSpeakerDiarization: total %{public}.3f s, audio "
        "%{public}.3f s, RTF %{public}.3f",
        total_seconds, audio_duration, total_seconds / audio_duration);
#else
    SHERPA_ONNX_LOGE(
        "OfflineSpeakerDiarization: total %.3f s, audio %.3f s, RTF %.3f",
        total_seconds, audio_duration, total_seconds / audio_duration);
#endif
  }

  void Init() {
    f104::Require(config_.segmentation.num_threads==4 && config_.embedding.num_threads==4 &&
        config_.segmentation.provider=="cpu" && config_.embedding.provider=="cpu" &&
        !config_.segmentation.debug && !config_.embedding.debug &&
        config_.clustering.num_clusters==-1 && config_.clustering.threshold==0.5f &&
        config_.min_duration_on==0.0f && config_.min_duration_off==0.0f,
        "UNREVIEWED_ALGORITHM_PARAMETERS");
    const auto &m = segmentation_model_.GetModelMetaData();
    f104::Require(m.sample_rate==16000 && m.window_size==160000 && m.window_shift==16000 &&
        m.receptive_field_size==991 && m.receptive_field_shift==270 &&
        m.num_speakers==3 && m.powerset_max_classes==2 && m.num_classes==7 &&
        embedding_extractor_.Dim()==512, "UNREVIEWED_GRAPH_METADATA"); InitPowersetMapping(); }

  // see also
  // https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/utils/powerset.py#L68
  void InitPowersetMapping() {
    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t num_classes = meta_data.num_classes;
    int32_t powerset_max_classes = meta_data.powerset_max_classes;
    int32_t num_speakers = meta_data.num_speakers;

    powerset_mapping_ = Matrix2DInt32(num_classes, num_speakers);
    powerset_mapping_.setZero();

    int32_t k = 1;
    for (int32_t i = 1; i <= powerset_max_classes; ++i) {
      if (i == 1) {
        for (int32_t j = 0; j != num_speakers; ++j, ++k) {
          powerset_mapping_(k, j) = 1;
        }
      } else if (i == 2) {
        for (int32_t j = 0; j != num_speakers; ++j) {
          for (int32_t m = j + 1; m < num_speakers; ++m, ++k) {
            powerset_mapping_(k, j) = 1;
            powerset_mapping_(k, m) = 1;
          }
        }
      } else {
#if __OHOS__
        SHERPA_ONNX_LOGE(
            "powerset_max_classes = %{public}d is currently not supported!", i);
#else
        SHERPA_ONNX_LOGE(
            "powerset_max_classes = %d is currently not supported!", i);
#endif
        SHERPA_ONNX_EXIT(-1);
      }
    }
  }

  std::vector<Matrix2D> RunSpeakerSegmentationModel(const float *audio,
                                                    int32_t n) const {
    std::vector<Matrix2D> ans;

    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t window_size = meta_data.window_size;
    int32_t window_shift = meta_data.window_shift;

    if (n <= 0) {
#if __OHOS__
      SHERPA_ONNX_LOGE(
          "number of audio samples is %{public}d (<= 0). Please provide a "
          "positive number",
          n);
#else
      SHERPA_ONNX_LOGE(
          "number of audio samples is %d (<= 0). Please provide a positive "
          "number",
          n);
#endif
      return {};
    }

    if (n <= window_size) {
      std::vector<float> buf(window_size);
      // NOTE: buf is zero initialized by default

      std::copy(audio, audio + n, buf.data());

      Matrix2D m = ProcessChunk(buf.data());

      ans.push_back(std::move(m));

      return ans;
    }

    int32_t num_chunks = (n - window_size) / window_shift + 1;
    bool has_last_chunk = ((n - window_size) % window_shift) > 0;

    ans.reserve(num_chunks + has_last_chunk);

    const float *p = audio;

    for (int32_t i = 0; i != num_chunks; ++i, p += window_shift) {
      Matrix2D m = ProcessChunk(p);

      ans.push_back(std::move(m));
    }

    if (has_last_chunk) {
      std::vector<float> buf(window_size);
      std::copy(p, audio + n, buf.data());

      Matrix2D m = ProcessChunk(buf.data());

      ans.push_back(std::move(m));
    }

    return ans;
  }

  Matrix2D ProcessChunk(const float *p) const {
    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t window_size = meta_data.window_size;

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> shape = {1, 1, window_size};

    Ort::Value x =
        Ort::Value::CreateTensor(memory_info, const_cast<float *>(p),
                                 window_size, shape.data(), shape.size());

    Ort::Value out = segmentation_model_.Forward(std::move(x));
    std::vector<int64_t> out_shape = out.GetTensorTypeAndShapeInfo().GetShape();
    f104::Require(out_shape==std::vector<int64_t>({1,589,7}), "MALFORMED_SEGMENTATION_SHAPE");
    f104::MatrixBytes(589,7);
    Matrix2D m(589,7);
    std::copy(out.GetTensorData<float>(), out.GetTensorData<float>() + m.size(),
              &m(0, 0));
    for (int64_t i=0;i<m.size();++i) f104::Require(std::isfinite(m.data()[i]), "NONFINITE_SEGMENTATION_SCORE");
    return m;
  }

  Matrix2DInt32 ToMultiLabel(const Matrix2D &m) const {
    int32_t num_rows = m.rows();
    Matrix2DInt32 ans(num_rows, powerset_mapping_.cols());

    std::ptrdiff_t col_id;

    for (int32_t i = 0; i != num_rows; ++i) {
      m.row(i).maxCoeff(&col_id);
      ans.row(i) = powerset_mapping_.row(col_id);
    }

    return ans;
  }

  // See also
  // https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/pipelines/utils/diarization.py#L122
  Int32RowVector ComputeSpeakersPerFrame(
      const std::vector<Matrix2DInt32> &labels) const {
    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t window_size = meta_data.window_size;
    int32_t window_shift = meta_data.window_shift;
    int32_t receptive_field_shift = meta_data.receptive_field_shift;

    int32_t num_chunks = labels.size();

    int32_t num_frames = f104::GlobalFrames(num_chunks,window_size,window_shift,receptive_field_shift);

    f104::MatrixBytes(1,num_frames);
    FloatRowVector count(num_frames);
    FloatRowVector weight(num_frames);
    count.setZero();
    weight.setZero();

    for (int32_t i = 0; i != num_chunks; ++i) {
      int32_t start = f104::FrameOffset(i,window_shift,receptive_field_shift);
      f104::Require(start+labels[i].rows()<=num_frames,"MALFORMED_RF_GRID");

      auto seq = Eigen::seqN(start, labels[i].rows());

      count(seq).array() += labels[i].rowwise().sum().array().cast<float>();

      weight(seq).array() += 1;
    }

    return ((count.array() / (weight.array() + 1e-12f)) + 0.5).cast<int32_t>();
  }

  void CaptureActivity(const std::vector<Matrix2DInt32> &labels,int32_t n,f104::Diagnostics *d) const {
    const auto &m=segmentation_model_.GetModelMetaData();
    for(int32_t c=0;c<static_cast<int32_t>(labels.size());++c) {
      auto local_n=std::min<int64_t>(m.window_size,n-c*m.window_shift);
      auto keep=f104::KeepFrames(labels[c].rows(),local_n,m.receptive_field_size,m.receptive_field_shift);
      for(int32_t slot=0;slot<m.num_speakers;++slot) {
        bool active=false;int64_t start=0;bool any=false;
        for(int64_t f=0;f<=keep;++f) {
          bool on=f<keep && labels[c](f,slot)!=0;
          if(on && !active){start=f;active=true;any=true;}
          if(!on && active) {
            auto span=f104::FrameSpan(start,f,keep,local_n,m.receptive_field_size,m.receptive_field_shift);
            span=f104::Clip2(span.start2+2LL*c*m.window_shift,span.end2+2LL*c*m.window_shift,n);
            f104::AddActivity(d,c,slot,-1,span);active=false;
          }
        }
        if(any)d->mappings.push_back({c,slot,-1,"NO_VALID_ISOLATED_EMBEDDING"});
      }
    }
  }

  std::pair<std::vector<Int32Pair>,std::vector<std::vector<Int32Pair>>>
  GetChunkSpeakerSampleIndexes(const std::vector<Matrix2DInt32> &labels,int32_t n) const {
    auto isolated=ExcludeOverlap(labels);
    std::vector<Int32Pair> pairs;std::vector<std::vector<Int32Pair>> spans;
    int64_t span_count=0;
    const auto &m=segmentation_model_.GetModelMetaData();
    for(int32_t c=0;c<static_cast<int32_t>(isolated.size());++c) {
      auto local_n=std::min<int64_t>(m.window_size,n-c*m.window_shift);
      auto keep=f104::KeepFrames(isolated[c].rows(),local_n,m.receptive_field_size,m.receptive_field_shift);
      for(int32_t slot=0;slot<m.num_speakers;++slot) {
        if(isolated[c].col(slot).sum()<10)continue; // versioned min isolated frames
        std::vector<Int32Pair> v;bool active=false;int64_t start=0;
        for(int64_t f=0;f<=keep;++f) {
          bool on=f<keep && isolated[c](f,slot)!=0;
          if(on && !active){start=f;active=true;}
          if(!on && active) {
            auto span=f104::FrameSpan(start,f,keep,local_n,m.receptive_field_size,m.receptive_field_shift);
            // Integer sample edges: outward rounding, clipped to observed PCM.
            int64_t a=c*m.window_shift+span.start2/2;
            int64_t b=c*m.window_shift+(span.end2+1)/2;
            a=std::clamp<int64_t>(a,0,n);b=std::clamp<int64_t>(b,0,n);
            if(a<b){
              f104::Require(span_count<f104::kTurns,"POPULATION_LIMIT: isolated spans");
              ++span_count;v.emplace_back(a,b);
            }
            active=false;
          }
        }
        if(!v.empty()){pairs.emplace_back(c,slot);spans.push_back(std::move(v));}
      }
    }
    f104::Require(pairs.size()<=static_cast<size_t>(f104::kRows),"POPULATION_LIMIT: embedding rows");
    return {pairs,spans};
  }

  // If there are multiple speakers at a frame, then this frame is excluded.
  std::vector<Matrix2DInt32> ExcludeOverlap(
      const std::vector<Matrix2DInt32> &labels) const {
    int32_t num_chunks = labels.size();
    std::vector<Matrix2DInt32> ans;
    ans.reserve(num_chunks);

    for (const auto &label : labels) {
      f104::MatrixBytes(label.rows(),label.cols());
      Matrix2DInt32 new_label(label.rows(), label.cols());
      new_label.setZero();
      Int32RowVector v = label.rowwise().sum();

      for (int32_t i = 0; i != v.cols(); ++i) {
        if (v[i] < 2) {
          new_label.row(i) = label.row(i);
        }
      }

      ans.push_back(std::move(new_label));
    }

    return ans;
  }

  /**
   * @param sample_indexes[i] contains the sample segment start and end indexes
   *                          for the i-th (chunk, speaker) pair
   * @return Return a matrix of shape (sample_indexes.size(), embedding_dim)
   *         where ans.row[i] contains the embedding for the
   *         i-th (chunk, speaker) pair
   */
  Matrix2D ComputeEmbeddings(const float *audio,int32_t n,const std::vector<Int32Pair>&pairs,
      const std::vector<std::vector<Int32Pair>>& indexes,std::vector<int32_t>*valid,f104::Diagnostics *d,
      OfflineSpeakerDiarizationProgressCallback callback,void *callback_arg) const {
    f104::Require(indexes.size()<=static_cast<size_t>(f104::kRows),"POPULATION_LIMIT: embedding rows");
    f104::MatrixBytes(indexes.size(),512);
    Matrix2D ans(indexes.size(),512);int32_t row=0,index=0;
    for(const auto &v:indexes) {
      auto stream=embedding_extractor_.CreateStream();
      for(const auto &p:v) {
        f104::Require(0<=p.first && p.first<p.second && p.second<=n,"MALFORMED_ISOLATED_SPAN");
        stream->AcceptWaveform(16000,audio+p.first,p.second-p.first);
      }
      stream->InputFinished();
      if(embedding_extractor_.IsReady(stream.get())) {
        auto embedding=embedding_extractor_.Compute(stream.get());
        const char *problem=f104::EmbeddingProblem(embedding.data(),embedding.size());
        if(problem==nullptr) {
          std::copy(embedding.begin(),embedding.end(),ans.row(row).data());
          valid->push_back(index);++row;
        } else {
          d->embedding_rejections.push_back({pairs[index].first,pairs[index].second,problem});
        }
      } else {
        d->embedding_rejections.push_back({pairs[index].first,pairs[index].second,"INSUFFICIENT_ISOLATED_SAMPLES"});
      }
      ++index;if(callback)callback(index,indexes.size(),callback_arg);
    }
    ans.conservativeResize(row,512);return ans;
  }


  std::unordered_map<Int32Pair, int32_t, PairHash> ConvertChunkSpeakerToCluster(
      const std::vector<Int32Pair> &chunk_speaker_pair,
      const std::vector<int32_t> &cluster_labels) const {
    std::unordered_map<Int32Pair, int32_t, PairHash> ans;

    f104::Require(chunk_speaker_pair.size()==cluster_labels.size(),"MALFORMED_MAPPING");
    int32_t k = 0;
    for (const auto &p : chunk_speaker_pair) {
      if (k >= static_cast<int32_t>(cluster_labels.size())) {
        SHERPA_ONNX_LOGE("cluster_labels size mismatch: k=%d, size=%d", k,
                         static_cast<int32_t>(cluster_labels.size()));
        break;
      }
      ans[p] = cluster_labels[k];
      k += 1;
    }

    return ans;
  }

  std::vector<Matrix2DInt32> ReLabel(
      const std::vector<Matrix2DInt32> &labels, int32_t max_cluster_index,
      std::unordered_map<Int32Pair, int32_t, PairHash> chunk_speaker_to_cluster)
      const {
    f104::Require(max_cluster_index>=0 && max_cluster_index<f104::kClusters,
                  "POPULATION_LIMIT: intermediate clusters");
    std::vector<Matrix2DInt32> new_labels;
    new_labels.reserve(labels.size());

    int32_t chunk_index = 0;
    for (const auto &label : labels) {
      f104::MatrixBytes(label.rows(),max_cluster_index+1);
      Matrix2DInt32 new_label(label.rows(), max_cluster_index + 1);
      new_label.setZero();

      Matrix2DInt32 t = label.transpose();
      // t: (num_speakers, num_frames)

      for (int32_t speaker_index = 0; speaker_index != t.rows();
           ++speaker_index) {
        if (chunk_speaker_to_cluster.count({chunk_index, speaker_index}) == 0) {
          continue;
        }

        int32_t new_speaker_index =
            chunk_speaker_to_cluster.at({chunk_index, speaker_index});

        if (new_speaker_index < 0 ||
            new_speaker_index >= new_label.cols()) {
          SHERPA_ONNX_LOGE(
              "ReLabel: new_speaker_index %d out of range [0, %d), skipping",
              new_speaker_index,
              static_cast<int32_t>(new_label.cols()));
          continue;
        }

        for (int32_t k = 0; k != t.cols(); ++k) {
          if (t(speaker_index, k) == 1) {
            new_label(k, new_speaker_index) = 1;
          }
        }
      }

      new_labels.push_back(std::move(new_label));

      chunk_index += 1;
    }

    return new_labels;
  }

  Matrix2DInt32 ComputeSpeakerCount(const std::vector<Matrix2DInt32> &labels,
                                    int32_t num_samples) const {
    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t window_size = meta_data.window_size;
    int32_t window_shift = meta_data.window_shift;
    int32_t receptive_field_shift = meta_data.receptive_field_shift;

    int32_t num_chunks = labels.size();

    int32_t num_frames = f104::GlobalFrames(num_chunks,window_size,window_shift,receptive_field_shift);

    f104::Require(!labels.empty(),"EMPTY_LABEL_POPULATION");
    f104::MatrixBytes(num_frames,labels[0].cols());
    Matrix2DInt32 count(num_frames, labels[0].cols());
    count.setZero();

    for (int32_t i = 0; i != num_chunks; ++i) {
      int32_t start = f104::FrameOffset(i,window_shift,receptive_field_shift);
      f104::Require(start+labels[i].rows()<=num_frames,"MALFORMED_RF_GRID");

      auto seq = Eigen::seqN(start, labels[i].rows());

      count(seq, Eigen::placeholders::all).array() += labels[i].array();
    }

    auto keep=f104::KeepFrames(count.rows(),num_samples,meta_data.receptive_field_size,receptive_field_shift);
    return count.topRows(keep);
  }


  Matrix2DInt32 FinalizeLabels(const Matrix2DInt32 &count,
                               const Int32RowVector &speakers_per_frame) const {
    int32_t num_rows = count.rows();
    int32_t num_cols = count.cols();

    f104::MatrixBytes(num_rows,num_cols);
    Matrix2DInt32 ans(num_rows, num_cols);
    ans.setZero();

    for (int32_t i = 0; i != num_rows; ++i) {
      int32_t k = std::min<int32_t>(speakers_per_frame[i],num_cols);
      if (k == 0) {
        continue;
      }
      auto top_k = TopkIndex(&count(i, 0), num_cols, k);

      for (int32_t m : top_k) {
        if (m >= 0 && m < num_cols && count(i,m)>0) {
          ans(i, m) = 1;
        }
      }
    }

    return ans;
  }

  OfflineSpeakerDiarizationResult ComputeResult(
      const Matrix2DInt32 &labels,int32_t n,f104::Diagnostics *d) const {
    OfflineSpeakerDiarizationResult ans;
    if(labels.rows()==0 || labels.cols()==0)return ans;
    const auto &m=segmentation_model_.GetModelMetaData();
    for(int32_t speaker=0;speaker<labels.cols();++speaker) {
      bool active=false;int64_t start=0;
      for(int64_t f=0;f<=labels.rows();++f) {
        bool on=f<labels.rows() && labels(f,speaker)>0;
        if(on && !active){active=true;start=f;}
        if(!on && active) {
          auto span=f104::FrameSpan(start,f,labels.rows(),n,m.receptive_field_size,m.receptive_field_shift);
          if(!span.empty && span.start_ms<span.end_ms) {
            f104::Require(d->turns.size()<static_cast<size_t>(f104::kTurns),"POPULATION_LIMIT: turns");
            d->turns.push_back({speaker,span});
            // Legacy float seconds are compatibility only; v2 consumes exact
            // twice-sample edges and inward bounded integer milliseconds.
            ans.Add(OfflineSpeakerDiarizationSegment(span.start2/32000.0f,span.end2/32000.0f,speaker));
          }
          active=false;
        }
      }
    }
    return ans;
  }


  void MergeSegments(
      std::vector<OfflineSpeakerDiarizationSegment> *segments) const {
    float min_duration_off = config_.min_duration_off;
    bool changed = true;
    while (changed) {
      changed = false;
      for (int32_t i = 0; i < static_cast<int32_t>(segments->size()) - 1; ++i) {
        auto s = (*segments)[i].Merge((*segments)[i + 1], min_duration_off);
        if (s) {
          (*segments)[i] = s.value();
          segments->erase(segments->begin() + i + 1);

          changed = true;
          break;
        }
      }
    }
  }

 private:
  OfflineSpeakerDiarizationConfig config_;
  OfflineSpeakerSegmentationPyannoteModel segmentation_model_;
  SpeakerEmbeddingExtractor embedding_extractor_;
  std::unique_ptr<FastClustering> clustering_;
  Matrix2DInt32 powerset_mapping_;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_PYANNOTE_IMPL_H_
