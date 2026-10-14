// F104 source-fix1, Apache-2.0. Unqualified standalone FD entry source.
// Not wired to any provider/host base and not built with neural dependencies.
#ifndef SHERPA_ONNX_CSRC_F104_NATIVE_FD_ENTRY_H_
#define SHERPA_ONNX_CSRC_F104_NATIVE_FD_ENTRY_H_
#include "sherpa-onnx/csrc/f104-diagnostic-fd.h"
#include "sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h"
namespace sherpa_onnx { namespace f104 {
// Implemented only by the already existing managed Content/License/execution
// owner after exact dependency, graph, carrier and tuple review. No command
// line/JSON/public constructor can create a production admission here.
class ManagedDevelopmentOwner {
 public:
  virtual ~ManagedDevelopmentOwner()=default;
  virtual void VerifyExactDevelopmentTuple(int pcm_fd,int segmentation_fd,
      int embedding_fd,int64_t samples,const char *algorithm) const=0;
  virtual void ProveOwnedTeardownOnException() const=0;
  virtual void VerifyCompletionStillOwned() const=0;
};
inline void RunDevelopmentFds(const ManagedDevelopmentOwner&owner,int pcm_fd,
    int segmentation_fd,int embedding_fd,int metadata_fd,int scores_fd) {
  Require(pcm_fd!=segmentation_fd && pcm_fd!=embedding_fd && segmentation_fd!=embedding_fd &&
      metadata_fd!=scores_fd && metadata_fd!=pcm_fd && metadata_fd!=segmentation_fd &&
      metadata_fd!=embedding_fd && scores_fd!=pcm_fd && scores_fd!=segmentation_fd && scores_fd!=embedding_fd,
      "DISTINCT_FDS_REQUIRED");
  uint64_t bytes=SealedSize(pcm_fd,kSamples*4);
  Require(bytes>0 && bytes%4==0,"MALFORMED_PCM_SIZE");int64_t n=bytes/4;
  SealedSize(segmentation_fd,16ULL<<20);SealedSize(embedding_fd,128ULL<<20);
  owner.VerifyExactDevelopmentTuple(pcm_fd,segmentation_fd,embedding_fd,n,kAlgorithm);
  Envelope(n,160000,16000,589,7,3,270); // before PCM/workspace allocation
  Diagnostics d;d.samples=n;
  try {
    std::vector<float> pcm(n);uint64_t position=0;
    while(position<bytes) {
      ssize_t read=pread(pcm_fd,reinterpret_cast<char*>(pcm.data())+position,bytes-position,position);
      if(read<0 && errno==EINTR)continue;
      Require(read>0,"TRUNCATED_PCM");position+=read;
    }
    ValidatePcm(pcm.data(),n);
    OfflineSpeakerDiarizationConfig c;
    // The legacy C-API maps zero merge/drop to defaults; do not use it here.
    c.segmentation.pyannote.model="/proc/self/fd/"+std::to_string(segmentation_fd);
    c.segmentation.pyannote.window_shift_ratio=0.1f;
    c.segmentation.num_threads=4;c.segmentation.provider="cpu";c.segmentation.debug=false;
    c.embedding.model="/proc/self/fd/"+std::to_string(embedding_fd);
    c.embedding.num_threads=4;c.embedding.provider="cpu";c.embedding.debug=false;
    c.clustering.num_clusters=-1;c.clustering.threshold=0.5f;
    c.min_duration_on=0;c.min_duration_off=0;
    { // models/ORT sessions destroyed before completion record is emitted
      OfflineSpeakerDiarizationPyannoteImpl pipeline(c);
      pipeline.ProcessDetailed(pcm.data(),n,&d);
    }
    owner.VerifyCompletionStillOwned();
    WriteDiagnosticsFds(d,metadata_fd,scores_fd);
  } catch(...) {
    owner.ProveOwnedTeardownOnException();
    // An exception is an attempt failure; no partial record becomes completion.
    throw;
  }
}
}} // namespace sherpa_onnx::f104
#endif
