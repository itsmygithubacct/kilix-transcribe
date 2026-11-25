// Local explicit-session diarization adapter, Apache-2.0.
// The provider verifies and seals the selected runtime/model population before
// its existing owned supervisor launches this one-job process.
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <time.h>
#include <unistd.h>
#include "sherpa-onnx/csrc/f104-native-fd-entry.h"
#include "sherpa-onnx/csrc/f104-canonical-features.h"
#include "sherpa-onnx/csrc/speaker-embedding-extractor-model.h"

namespace {
double Now() {
  timespec value{};
  if (clock_gettime(CLOCK_MONOTONIC, &value)) throw std::runtime_error("clock");
  return value.tv_sec + value.tv_nsec / 1000000000.0;
}
int Number(const char *value) {
  char *end = nullptr;
  errno = 0;
  long number = std::strtol(value, &end, 10);
  if (errno || !end || *end || end == value || number < 0 || number > INT_MAX)
    throw std::runtime_error("descriptor");
  return static_cast<int>(number);
}
class Owner final : public sherpa_onnx::f104::ManagedDevelopmentOwner {
 public:
  Owner(int parent, double deadline) : parent_(parent), deadline_(deadline) { Check(); }
  void VerifyExactDevelopmentTuple(int, int, int, int64_t, const char *algorithm) const override {
    Check();
    if (std::strcmp(algorithm, "f104-diarization-algorithm/2.1"))
      throw std::runtime_error("algorithm");
  }
  void ProveOwnedTeardownOnException() const override {
    // This adapter spawns no descendants. C++ stack unwinding destroys its
    // sessions; the provider supervisor proves process-tree teardown on exit.
  }
  void VerifyCompletionStillOwned() const override { Check(); }
 private:
  void Check() const {
    if (parent_ <= 1 || getppid() != parent_ || !std::isfinite(deadline_)
        || Now() >= deadline_) throw std::runtime_error("owner/deadline");
  }
  int parent_;
  double deadline_;
};
void Write(int descriptor, const void *data, size_t bytes) {
  const char *source = static_cast<const char *>(data);
  while (bytes) {
    ssize_t count = write(descriptor, source, bytes);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) throw std::runtime_error("output");
    source += count; bytes -= count;
  }
}
void Features(const Owner &owner, int pcm, int feature_output, int embedding, int embedding_output) {
  using namespace sherpa_onnx;
  uint64_t bytes = f104::SealedSize(pcm, f104::kSamples * 4);
  if (bytes % 4 || bytes < 400 * 4) throw std::runtime_error("waveform");
  std::vector<float> samples(bytes / 4);
  size_t position = 0;
  while (position < bytes) {
    ssize_t count = pread(pcm, reinterpret_cast<char *>(samples.data()) + position, bytes-position, position);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) throw std::runtime_error("waveform");
    position += count;
  }
  f104::ValidatePcm(samples.data(), samples.size());
  f104::SealedSize(embedding, 128ULL << 20);
  FeatureExtractorConfig settings;
  f104::EnglishFeatures(&settings);
  FeatureExtractor extractor(settings);
  extractor.AcceptWaveform(16000, samples.data(), samples.size());
  extractor.InputFinished();
  int frames = extractor.NumFramesReady();
  f104::Require(frames > 0 && frames <= 360000, "FEATURE_FRAME_LIMIT");
  auto features = extractor.GetFrames(0, frames);
  f104::Require(features.size() == static_cast<size_t>(frames)*80, "FEATURE_SHAPE");
  auto matrix = Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>(
      features.data(), frames, 80);
  matrix = matrix.rowwise() - matrix.colwise().mean();
  SpeakerEmbeddingExtractorConfig config;
  config.model = "/proc/self/fd/" + std::to_string(embedding);
  config.num_threads = 4; config.provider = "cpu"; config.debug = false;
  SpeakerEmbeddingExtractorModel model(config);
  const auto &meta = model.GetMetaData();
  f104::Require(meta.sample_rate == 16000 && meta.normalize_samples && meta.output_dim == 512
      && meta.language == "en" && meta.feature_normalize_type == "global-mean", "EMBEDDING_METADATA");
  auto memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator,OrtMemTypeDefault);
  std::array<int64_t,3> shape{1,frames,80};
  auto tensor = Ort::Value::CreateTensor(memory,features.data(),features.size(),shape.data(),shape.size());
  auto output = model.Compute(std::move(tensor));
  f104::Require(output.GetTensorTypeAndShapeInfo().GetShape() == std::vector<int64_t>({1,512}),
                 "EMBEDDING_SHAPE");
  owner.VerifyCompletionStillOwned();
  Write(feature_output,features.data(),features.size()*4);
  Write(embedding_output,output.GetTensorData<float>(),512*4);
}
}

int main(int argc, char **argv) {
  try {
    if (argc != 8) throw std::runtime_error("arguments");
    char *end = nullptr;
    double deadline = std::strtod(argv[7], &end);
    if (!end || end == argv[7] || *end) throw std::runtime_error("deadline");
    Owner owner(Number(argv[6]), deadline);
    if (!std::strcmp(argv[1], "--features")) {
      Features(owner,Number(argv[2]),Number(argv[3]),Number(argv[4]),Number(argv[5]));
      return 0;
    }
    sherpa_onnx::f104::RunDevelopmentFds(owner, Number(argv[1]), Number(argv[2]),
        Number(argv[3]), Number(argv[4]), Number(argv[5]));
    return 0;
  } catch (...) {
    // No source paths, waveform, transcript or raw neural exception crosses
    // the product boundary. Failure never commits a partial annotation.
    std::cerr << "KILIX_DIARIZATION_ENGINE_FAILED\n";
    return 70;
  }
}
