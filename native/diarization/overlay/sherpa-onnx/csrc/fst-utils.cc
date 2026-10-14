// F104 native build compatibility modification, 2026-10-09: owned sealed FAR buffer.
// F104 native-build-fix1 draft, 2026-10-09: exception-safe graph ownership.
// sherpa-onnx/csrc/fst-utils.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/fst-utils.h"
#include "sherpa-onnx/csrc/far-buffer-reader.h"

#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "fst/extensions/far/far.h"
#include "kaldifst/csrc/kaldi-fst-io.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

// This function is copied from kaldi.
//
// @param filename Path to a StdVectorFst or StdConstFst graph
// @return The caller should free the returned pointer using `delete` to
//         avoid memory leak.
fst::Fst<fst::StdArc> *ReadGraph(const std::string &filename) {
  // read decoding network FST
  auto is = OpenInputFile(filename, std::ios::binary);
  if (!is.good()) {
    SHERPA_ONNX_LOGE("Could not open decoding-graph FST %s", filename.c_str());
  }

  fst::FstHeader hdr;
  if (!hdr.Read(is, "<unknown>")) {
    SHERPA_ONNX_LOGE("Reading FST: error reading FST header.");
  }

  if (hdr.ArcType() != fst::StdArc::Type()) {
    SHERPA_ONNX_LOGE("FST with arc type %s not supported",
                     hdr.ArcType().c_str());
  }
  fst::FstReadOptions ropts("<unspecified>", &hdr);

  fst::Fst<fst::StdArc> *decode_fst = nullptr;

  if (hdr.FstType() == "vector") {
    decode_fst = fst::VectorFst<fst::StdArc>::Read(is, ropts);
  } else if (hdr.FstType() == "const") {
    decode_fst = fst::ConstFst<fst::StdArc>::Read(is, ropts);
  } else {
    SHERPA_ONNX_LOGE("Reading FST: unsupported FST type: %s",
                     hdr.FstType().c_str());
  }

  if (decode_fst == nullptr) {  // fst code will warn.
    SHERPA_ONNX_LOGE("Error reading FST (after reading header).");
    return nullptr;
  } else {
    return decode_fst;
  }
}

std::vector<std::unique_ptr<fst::StdConstFst>> ReadFstsFromFar(
    const std::vector<char> &buffer) {
  std::vector<std::unique_ptr<fst::StdConstFst>> ans;

  FarBufferReader owner(buffer);
  auto& reader = owner.reader();
  for (; !reader.Done(); reader.Next()) {
    const auto* graph = reader.GetFst();
    if (!graph) throw std::runtime_error("FAR_READER_ENTRY_FAILED");
    std::unique_ptr<fst::Fst<fst::StdArc>> copy(graph->Copy());
    if (!copy) throw std::runtime_error("FAR_READER_COPY_FAILED");
    // The exact pinned converter consumes a vector copy on success or returns
    // the same const copy. It leaves input intact if construction throws.
    std::unique_ptr<fst::StdConstFst> retained(
        fst::CastOrConvertToConstFst(copy.get()));
    if (!retained) throw std::runtime_error("FAR_READER_CONVERSION_FAILED");
    copy.release();  // Success transferred/consumed it; these steps cannot throw.
    ans.emplace_back(std::move(retained));
  }
  if (reader.Error()) throw std::runtime_error("FAR_READER_READ_FAILED");

  return ans;
}

}  // namespace sherpa_onnx
