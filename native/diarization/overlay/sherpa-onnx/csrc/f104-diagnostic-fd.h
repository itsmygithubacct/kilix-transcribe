// F104 source-fix1, Apache-2.0. Descriptor-only diagnostic producer.
// F104 source-fix2, 2026-10-09: complete validation before either FD write.
#ifndef SHERPA_ONNX_CSRC_F104_DIAGNOSTIC_FD_H_
#define SHERPA_ONNX_CSRC_F104_DIAGNOSTIC_FD_H_
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include "sherpa-onnx/csrc/f104-diarization-guard.h"
#include "sherpa-onnx/csrc/f104-diagnostic-validation.h"
namespace sherpa_onnx { namespace f104 {
constexpr int kSeals=F_SEAL_WRITE|F_SEAL_SHRINK|F_SEAL_GROW|F_SEAL_SEAL;
constexpr uint64_t kMetadataBytes=16ULL<<20;
inline uint64_t SealedSize(int fd,uint64_t cap) {
  struct stat s{};Require(fd>=0 && fstat(fd,&s)==0 && S_ISREG(s.st_mode),"SEALED_FD_REQUIRED");
  int seals=fcntl(fd,F_GET_SEALS);
  Require(seals>=0 && (seals&kSeals)==kSeals && s.st_size>=0 && uint64_t(s.st_size)<=cap,
          "SEALED_FD_SIZE_OR_IMMUTABILITY");
  return s.st_size;
}
inline void EmptyOutput(int fd) {
  struct stat s{};Require(fd>=0 && fstat(fd,&s)==0 && S_ISREG(s.st_mode) && s.st_size==0,
                         "EMPTY_OUTPUT_FD_REQUIRED");
  Require(fcntl(fd,F_GET_SEALS)==0,"OWNED_UNSEALED_OUTPUT_REQUIRED");
}
inline void WriteAll(int fd,const void *data,uint64_t size) {
  auto p=static_cast<const char*>(data);
  while(size) {
    ssize_t n=write(fd,p,size);
    if(n<0 && errno==EINTR)continue;
    Require(n>0,"OUTPUT_WRITE_FAILED");p+=n;size-=n;
  }
}
class Json {
 public:
  std::string bytes;
  void Add(const std::string&s){Require(s.size()<=kMetadataBytes-bytes.size(),"POPULATION_LIMIT: diagnostic metadata");bytes+=s;}
  void Q(const std::string&s){Require(s.size()<=128,"DIAGNOSTIC_STRING_LIMIT");Add("\"");
    for(unsigned char c:s){Require(c>=32 && c<127,"DIAGNOSTIC_ASCII_REQUIRED");if(c=='"'||c=='\\')Add("\\");Add(std::string(1,c));}Add("\"");}
  void N(int64_t v){Add(std::to_string(v));}
  void D(double v){Require(std::isfinite(v),"NONFINITE_DIAGNOSTIC_SOURCE");std::ostringstream s;s<<std::setprecision(17)<<v;Add(s.str());}
  void SpanValue(const Span&s){Require(!s.empty && s.start2<s.end2,"EMPTY_DIAGNOSTIC_SPAN");
    Add("\"start2\":");N(s.start2);Add(",\"end2\":");N(s.end2);Add(",\"start_ms\":");N(s.start_ms);Add(",\"end_ms\":");N(s.end_ms);}
};
// Scores are exact finite float32 little-endian bytes on a distinct sealed FD;
// no JSON expansion of a whole neural score plane. The receiving trusted
// adapter hashes BOTH immutable FDs and joins them to the job/source tuple.
inline void WriteDiagnosticsFds(const Diagnostics&d,int metadata_fd,int scores_fd) {
  Require(metadata_fd!=scores_fd,"DISTINCT_OUTPUT_FDS_REQUIRED");EmptyOutput(metadata_fd);EmptyOutput(scores_fd);
  ValidateDiagnostics(d);  // Complete validation before writing either FD.
  Require(d.state=="DEVELOPMENT" && d.schema=="f104.diarization.diagnostics/2.1", "UNADMITTED_CALIBRATION_STATE");
  Require(d.samples>0 && d.samples<=kSamples && d.scores.size()<=size_t(kChunks) &&
      d.mappings.size()<=size_t(kRows) && d.assignments.size()<=size_t(kRows) &&
      d.embedding_rejections.size()<=size_t(kRows) && d.activity.size()<=size_t(kTurns) &&
      d.turns.size()<=size_t(kTurns),"POPULATION_LIMIT: diagnostics");
  Require(d.status=="OK"||d.status=="SILENCE"||d.status=="PARTIAL_UNKNOWN"||
      d.status=="UNIDENTIFIABLE_ACTIVITY"||d.status=="FAILURE","INVALID_DIAGNOSTIC_STATUS");
  uint32_t endian=1;Require(*reinterpret_cast<unsigned char*>(&endian)==1 && sizeof(float)==4,
                           "LITTLE_ENDIAN_FLOAT32_REQUIRED");
  uint64_t score_bytes=0;Json j;
  j.Add("{\"schema\":");j.Q(d.schema);j.Add(",\"algorithm\":");j.Q(kAlgorithm);
  j.Add(",\"state\":\"DEVELOPMENT\",\"sample_rate\":16000,\"samples\":");j.N(d.samples);
  j.Add(",\"status\":");j.Q(d.status);j.Add(",\"error_code\":");
  if(d.error_code.empty())j.Add("null");else j.Q(d.error_code);
  j.Add(",\"score_domain\":");j.Q(d.score_domain);j.Add(",\"scores\":[");
  bool comma=false;
  for(const auto&s:d.scores) {
    Require(s.chunk>=0 && s.chunk<kChunks && s.offset==s.chunk*16000 && s.frames==589 && s.classes==7,
            "MALFORMED_SCORE_DESCRIPTOR");
    auto bytes=MatrixBytes(s.frames,s.classes);
    Require(s.values.size()==size_t(s.frames*s.classes),"MALFORMED_SCORE_POPULATION");
    for(auto x:s.values)Require(std::isfinite(x),"NONFINITE_SEGMENTATION_SCORE");
    if(comma){j.Add(",");}
    comma=true;
    j.Add("{\"chunk\":");j.N(s.chunk);j.Add(",\"offset_samples\":");j.N(s.offset);
    j.Add(",\"frames\":");j.N(s.frames);j.Add(",\"classes\":");j.N(s.classes);
    j.Add(",\"byte_offset\":");j.N(score_bytes);j.Add(",\"byte_count\":");j.N(bytes);j.Add("}");
    score_bytes=Add(score_bytes,bytes,kDiagnosticBytes);WriteAll(scores_fd,s.values.data(),bytes);
  }
  Require(score_bytes==d.score_bytes,"SCORE_BYTE_COUNT_MISMATCH");
  j.Add("],\"score_bytes\":");j.N(score_bytes);j.Add(",\"mappings\":[");comma=false;
  for(const auto&m:d.mappings){if(comma)j.Add(",");comma=true;j.Add("{\"chunk\":");j.N(m.chunk);
    j.Add(",\"slot\":");j.N(m.slot);j.Add(",\"cluster\":");if(m.cluster<0)j.Add("null");else j.N(m.cluster);
    j.Add(",\"reason\":");j.Q(m.reason);j.Add("}");}
  j.Add("],\"embedding_rejections\":[");comma=false;
  for(const auto&r:d.embedding_rejections){if(comma)j.Add(",");comma=true;j.Add("{\"chunk\":");j.N(r.chunk);
    j.Add(",\"slot\":");j.N(r.slot);j.Add(",\"reason\":");j.Q(r.reason);j.Add("}");}
  j.Add("],\"assignments\":[");comma=false;
  for(const auto&a:d.assignments){if(comma)j.Add(",");comma=true;j.Add("{\"chunk\":");j.N(a.chunk);
    j.Add(",\"slot\":");j.N(a.slot);j.Add(",\"cluster\":");j.N(a.cluster);
    j.Add(",\"assigned_cosine_distance\":");if(a.has_assigned_distance)j.D(a.assigned_cosine_distance);else j.Add("null");
    j.Add(",\"alternative_cosine_distance\":");if(a.has_margin)j.D(a.alternative_cosine_distance);else j.Add("null");
    j.Add(",\"margin\":");if(a.has_margin)j.D(a.margin);else j.Add("null");j.Add("}");}
  j.Add("],\"activity\":[");comma=false;
  for(const auto&a:d.activity){if(comma)j.Add(",");comma=true;j.Add("{\"chunk\":");j.N(a.chunk);
    j.Add(",\"slot\":");j.N(a.slot);j.Add(",\"cluster\":");if(a.cluster<0)j.Add("null");else j.N(a.cluster);
    j.Add(",");j.SpanValue(a.span);j.Add("}");}
  j.Add("],\"turns\":[");comma=false;
  for(const auto&t:d.turns){if(comma)j.Add(",");comma=true;j.Add("{\"cluster\":");j.N(t.cluster);
    j.Add(",");j.SpanValue(t.span);j.Add("}");}
  j.Add("],\"confidence\":null,\"calibration_binding\":null}\n");
  WriteAll(metadata_fd,j.bytes.data(),j.bytes.size());
  Require(fcntl(scores_fd,F_ADD_SEALS,kSeals)==0 && fcntl(metadata_fd,F_ADD_SEALS,kSeals)==0,"OUTPUT_SEAL_FAILED");
}
}} // namespace sherpa_onnx::f104
#endif
