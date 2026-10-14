// F104 source-fix1, Apache-2.0. Only the candidate's two model constructors use it.
#ifndef SHERPA_ONNX_CSRC_F104_CPU_SESSION_H_
#define SHERPA_ONNX_CSRC_F104_CPU_SESSION_H_
#include "sherpa-onnx/csrc/f104-diarization-guard.h"
#include "sherpa-onnx/csrc/session.h"
namespace sherpa_onnx { namespace f104 {
inline Ort::SessionOptions CpuSessionOptions(int32_t threads,const std::string&provider) {
  Require(threads==4 && provider=="cpu","UNREVIEWED_CPU_THREAD_SETTINGS");
  auto opts=GetSessionOptions(4,"cpu");
  opts.SetIntraOpNumThreads(4);opts.SetInterOpNumThreads(1);
  opts.SetExecutionMode(ORT_SEQUENTIAL);
  opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
  opts.AddConfigEntry("session.intra_op.allow_spinning","0");
  opts.AddConfigEntry("session.inter_op.allow_spinning","0");
  // Two retained pools can have more than four live threads. Effective CPU
  // entitlement is bounded by the existing owner/cgroup; measure all threads.
  return opts;
}
}} // namespace sherpa_onnx::f104
#endif
