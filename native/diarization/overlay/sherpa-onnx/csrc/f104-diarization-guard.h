// F104 development derivative, 2026-10-09. Apache-2.0; see notices.
// F104 source-fix2: validate FrameSpan samples before final-cell arithmetic.
// Synthetic controls exercise this source; neural equivalence/fit is unproved.
#ifndef SHERPA_ONNX_CSRC_F104_DIARIZATION_GUARD_H_
#define SHERPA_ONNX_CSRC_F104_DIARIZATION_GUARD_H_
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sherpa_onnx { namespace f104 {
constexpr const char *kAlgorithm = "f104-diarization-algorithm/2.1";
constexpr int64_t kSamples = 57600000, kChunks = 4096, kRows = 12000;
constexpr int64_t kClusters = 16, kGlobalFrames = 262144, kTurns = 16384;
constexpr uint64_t kMatrixBytes = 256ULL << 20, kDistanceBytes = 576000000;
constexpr uint64_t kDiagnosticBytes = 128ULL << 20, kPlannedWorkspace = 2ULL << 30;

inline void Require(bool valid, const char *reason) {
  if (!valid) throw std::runtime_error(reason);
}
inline uint64_t Mul(uint64_t a, uint64_t b, uint64_t cap) {
  Require(b == 0 || a <= cap / b, "POPULATION_LIMIT: checked multiplication");
  return a * b;
}
inline uint64_t Add(uint64_t a, uint64_t b, uint64_t cap) {
  Require(b <= cap && a <= cap - b, "POPULATION_LIMIT: checked addition");
  return a + b;
}
inline uint64_t MatrixBytes(int64_t rows, int64_t cols, uint64_t width=4,
                            uint64_t cap=kMatrixBytes) {
  Require(rows >= 0 && cols >= 0, "MALFORMED_DIMENSION");
  return Mul(Mul(static_cast<uint64_t>(rows), static_cast<uint64_t>(cols), cap), width, cap);
}
inline int64_t Chunks(int64_t n, int64_t window, int64_t shift) {
  Require(n > 0 && n <= kSamples && window > 0 && shift > 0 && shift <= window,
          "MALFORMED_GEOMETRY");
  int64_t count = n <= window ? 1 : 1 + (n-window)/shift + ((n-window)%shift != 0);
  Require(count <= kChunks, "POPULATION_LIMIT: chunks");
  return count;
}
inline uint64_t DistanceCount(int64_t rows) {
  Require(rows > 0 && rows <= kRows, "POPULATION_LIMIT: embedding rows");
  uint64_t a = static_cast<uint64_t>(rows), b = a - 1;
  if ((a & 1U) == 0) a /= 2; else b /= 2;
  return Mul(a, b, kDistanceBytes / sizeof(double));
}
inline void PositiveRows(int64_t rows, int64_t cols) {
  Require(rows > 0 && rows <= kRows && cols == 512, "INVALID_EMBEDDING_POPULATION");
  MatrixBytes(rows, cols);
}
inline const char *EmbeddingProblem(const float *row, int64_t dim) {
  if (!row || dim != 512) return "MALFORMED_EMBEDDING_SHAPE";
  double norm = 0;
  for (int64_t i=0;i<dim;++i) {
    if (!std::isfinite(row[i])) return "NONFINITE_EMBEDDING";
    norm = std::hypot(norm, static_cast<double>(row[i]));
  }
  return std::isfinite(norm) && norm > 1e-12 ? nullptr : "ZERO_OR_DEGENERATE_EMBEDDING";
}
inline bool ValidEmbedding(const float *row,int64_t dim) {
  return EmbeddingProblem(row,dim)==nullptr;
}
inline void Normalize(float *row, int64_t dim) {
  Require(ValidEmbedding(row,dim), "INVALID_EMBEDDING: finite nonzero norm required");
  double norm=0;
  for (int64_t i=0;i<dim;++i) norm=std::hypot(norm,static_cast<double>(row[i]));
  for (int64_t i=0;i<dim;++i) row[i]=static_cast<float>(row[i]/norm);
}
inline int64_t GlobalFrames(int64_t chunks, int64_t window, int64_t shift, int64_t rf_shift) {
  Require(chunks>0 && chunks<=kChunks && window==160000 && shift==16000 && rf_shift==270,
          "MALFORMED_GEOMETRY");
  uint64_t offset=Mul(chunks-1,shift,kSamples);
  // Union of actual 589-row chunk outputs, aligned to nearest global RF cell.
  // Window/RF-shift division invents unused trailing cells and drops end-active
  // turns. The final retained real cell owns the remaining observed tail.
  int64_t frames=static_cast<int64_t>((offset+rf_shift/2)/rf_shift+589);
  Require(frames<=kGlobalFrames,"POPULATION_LIMIT: global frames");
  return frames;
}
// Twice-sample Voronoi boundaries around RF centers. The first cell begins
// at zero; the final retained cell ends at the authoritative observed sample.
inline int64_t Boundary2(int64_t frame, int64_t rf_size, int64_t rf_shift) {
  Require(frame>=0 && frame<=kGlobalFrames && rf_size>0 && rf_size<=160000 &&
          rf_shift>0 && rf_shift<=160000,"MALFORMED_TIME_GEOMETRY");
  return frame==0 ? 0 : rf_size+(2*frame-1)*rf_shift;
}
inline int64_t KeepFrames(int64_t available, int64_t n, int64_t rf_size, int64_t rf_shift) {
  Require(available>=0 && available<=kGlobalFrames && n>=0 && n<=kSamples,
          "MALFORMED_TIME_POPULATION");
  if (n==0 || available==0) return 0;
  int64_t keep=1;
  while (keep<available && Boundary2(keep,rf_size,rf_shift)<2*n) ++keep;
  return keep;
}
struct Span {
  int64_t start2=0,end2=0,start_ms=0,end_ms=0;
  bool empty=true;
};
inline Span Clip2(int64_t start2,int64_t end2,int64_t n,int64_t rate=16000) {
  Require(n>=0 && n<=kSamples && rate==16000 && start2<=end2,"MALFORMED_TIME_SOURCE");
  Span s;
  s.start2=std::clamp<int64_t>(start2,0,2*n);
  s.end2=std::clamp<int64_t>(end2,0,2*n);
  s.empty=s.start2>=s.end2;
  if (s.empty) return s;
  int64_t divisor=2*rate;
  // Millisecond projection is inward, so it cannot extend past observed PCM.
  // A sub-ms interval remains in exact twice-sample diagnostics but is not a
  // renderable millisecond turn. Empty projections are discarded as turns.
  s.start_ms=(s.start2*1000+divisor-1)/divisor;
  s.end_ms=(s.end2*1000)/divisor;
  int64_t duration_ms=(n*1000)/rate;
  s.start_ms=std::clamp<int64_t>(s.start_ms,0,duration_ms);
  s.end_ms=std::clamp<int64_t>(s.end_ms,0,duration_ms);
  if(s.end_ms<s.start_ms)s.end_ms=s.start_ms;
  return s;
}
inline Span FrameSpan(int64_t start,int64_t end,int64_t frames,int64_t n,
                       int64_t rf_size,int64_t rf_shift) {
  Require(n>=0 && n<=kSamples,"MALFORMED_TIME_POPULATION");
  Require(0<=start && start<end && end<=frames && frames<=kGlobalFrames,"MALFORMED_FRAME_SOURCE");
  return Clip2(Boundary2(start,rf_size,rf_shift),end==frames ? 2*n : Boundary2(end,rf_size,rf_shift),n);
}
inline Span ClipSeconds(double start,double end,int64_t n) {
  Require(std::isfinite(start) && std::isfinite(end) && start<=end,"NONFINITE_TIME_SOURCE");
  Require(n>=0 && n<=kSamples,"MALFORMED_TIME_POPULATION");
  start=std::clamp(start,0.0,n/16000.0);
  end=std::clamp(end,0.0,n/16000.0);
  return Clip2(static_cast<int64_t>(std::floor(start*32000)),
               static_cast<int64_t>(std::ceil(end*32000)),n);
}
struct ChunkScores { int64_t chunk=0,offset=0,frames=0,classes=0;std::vector<float> values; };
struct Mapping { int32_t chunk=0,slot=0,cluster=-1;std::string reason; };
struct Assignment {
  int32_t chunk=0,slot=0,cluster=-1;
  double assigned_cosine_distance=0,alternative_cosine_distance=0,margin=0;
  bool has_assigned_distance=false,has_margin=false;
};
struct Activity { int32_t chunk=0,slot=0,cluster=-1;Span span; };
struct Turn { int32_t cluster=-1;Span span; };
struct Rejection { int32_t chunk=0,slot=0;std::string reason; };
struct Diagnostics {
  // No API can promote these diagnostics to calibrated confidence on its own.
  std::string schema="f104.diarization.diagnostics/2.1",state="DEVELOPMENT",status="NOT_RUN";
  std::string score_domain="FINITE_NEURAL_SCORES_UNVERIFIED_DOMAIN";
  std::string error_code;
  int64_t samples=0;uint64_t score_bytes=0;
  std::vector<ChunkScores> scores;
  std::vector<Mapping> mappings;
  std::vector<Assignment> assignments;
  std::vector<Activity> activity;
  std::vector<Turn> turns;
  std::vector<Rejection> embedding_rejections;
};
inline void AddActivity(Diagnostics *d,int32_t chunk,int32_t slot,int32_t cluster,Span span) {
  if (span.empty) return;
  Require(d->activity.size()<static_cast<size_t>(kTurns),"POPULATION_LIMIT: diagnostic activity");
  d->activity.push_back({chunk,slot,cluster,span});
}
inline void GuardClusterLabels(const std::vector<int32_t>&labels,int64_t rows) {
  Require(static_cast<int64_t>(labels.size())==rows && rows>0,"MALFORMED_CLUSTER_LABELS");
  for (auto label:labels) Require(label>=0 && label<kClusters,"POPULATION_LIMIT: intermediate clusters");
}
inline int64_t FrameOffset(int64_t chunk,int64_t shift,int64_t rf_shift) {
  Require(chunk>=0 && chunk<kChunks && shift==16000 && rf_shift==270,"MALFORMED_FRAME_OFFSET");
  return (chunk*shift+rf_shift/2)/rf_shift;
}
inline void ValidatePcm(const float *audio,int64_t n) {
  Require(audio && n>0 && n<=kSamples,"MALFORMED_PCM");
  for(int64_t i=0;i<n;++i)
    Require(std::isfinite(audio[i]) && std::abs(audio[i])<=1.0f,"NONFINITE_OR_UNNORMALIZED_PCM");
}
// Distances describe actual complete-link assignments; a centroid need not
// be nearest. Negative margins are evidence, not probabilities. K=1 has no
// alternative. No calibrated confidence is manufactured here.
inline std::vector<Assignment> AssignmentEvidence(const float *rows,int64_t count,
    const std::vector<int32_t>&labels,const std::vector<std::pair<int32_t,int32_t>>&pairs) {
  PositiveRows(count,512); GuardClusterLabels(labels,count);
  Require(rows && pairs.size()==static_cast<size_t>(count),"MALFORMED_ASSIGNMENTS");
  std::vector<std::vector<double>> centers(kClusters,std::vector<double>(512,0));
  std::vector<bool> present(kClusters,false),usable(kClusters,false);
  for(int64_t i=0;i<count;++i) {
    Require(ValidEmbedding(rows+i*512,512),"INVALID_EMBEDDING");
    double norm=0;for(int j=0;j<512;++j)norm=std::hypot(norm,double(rows[i*512+j]));
    present[labels[i]]=true;
    for(int j=0;j<512;++j)centers[labels[i]][j]+=rows[i*512+j]/norm;
  }
  for(int k=0;k<kClusters;++k)if(present[k]) {
    double norm=0;for(auto x:centers[k])norm=std::hypot(norm,x);
    usable[k]=std::isfinite(norm)&&norm>1e-12;
    if(usable[k])for(auto &x:centers[k])x/=norm;
  }
  std::vector<Assignment> ans;ans.reserve(count);
  for(int64_t i=0;i<count;++i) {
    Assignment a;a.chunk=pairs[i].first;a.slot=pairs[i].second;a.cluster=labels[i];
    double norm=0;for(int j=0;j<512;++j)norm=std::hypot(norm,double(rows[i*512+j]));
    double alternative=std::numeric_limits<double>::infinity();
    bool unusable_alternative=false;
    for(int k=0;k<kClusters;++k)if(present[k]) {
      if(!usable[k]) {if(k!=a.cluster)unusable_alternative=true;continue;}
      double dot=0;for(int j=0;j<512;++j)dot+=(rows[i*512+j]/norm)*centers[k][j];
      double distance=std::clamp(1.0-dot,0.0,2.0);
      if(k==a.cluster){a.assigned_cosine_distance=distance;a.has_assigned_distance=true;}
      else alternative=std::min(alternative,distance);
    }
    if(a.has_assigned_distance && std::isfinite(alternative) && !unusable_alternative) {
      a.alternative_cosine_distance=alternative;
      a.margin=alternative-a.assigned_cosine_distance;a.has_margin=true;
    }
    ans.push_back(a);
  }
  return ans;
}
// Admission-side checked worst envelope, not an RSS measurement. Libraries,
// ORT tensors and all allocator/transient copies still need cgroup/fit evidence.
inline void Envelope(int64_t n,int64_t window,int64_t shift,int64_t frames,
                       int64_t classes,int64_t slots,int64_t rf_shift) {
  Require(window==160000 && shift==16000 && frames==589 && classes==7 && slots==3 && rf_shift==270,
          "UNREVIEWED_GRAPH_GEOMETRY");
  auto chunks=Chunks(n,window,shift);
  auto globals=GlobalFrames(chunks,window,shift,rf_shift);
  uint64_t rows=Mul(chunks,slots,kRows);
  uint64_t total=Mul(n,4,kPlannedWorkspace);
  // Raw neural matrices plus retained diagnostic copy coexist during capture.
  total=Add(total,Mul(Mul(chunks,MatrixBytes(frames,classes),kDiagnosticBytes),2,kPlannedWorkspace),kPlannedWorkspace);
  total=Add(total,Mul(Mul(chunks,MatrixBytes(frames,slots),kMatrixBytes),2,kPlannedWorkspace),kPlannedWorkspace);
  total=Add(total,Mul(chunks,MatrixBytes(frames,kClusters),kMatrixBytes),kPlannedWorkspace);
  total=Add(total,Mul(DistanceCount(rows),sizeof(double),kDistanceBytes),kPlannedWorkspace);
  total=Add(total,Mul(MatrixBytes(rows,512),2,kPlannedWorkspace),kPlannedWorkspace);
  total=Add(total,Mul(MatrixBytes(globals,kClusters),3,kPlannedWorkspace),kPlannedWorkspace);
  total=Add(total,16ULL<<20,kPlannedWorkspace); // bounded metadata/transient trace
  (void)total;
}
}}  // namespace sherpa_onnx::f104
#endif
