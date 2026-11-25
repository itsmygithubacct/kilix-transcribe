// F104 source-fix2, Apache-2.0. Frozen score/activity/turn contract validation.
// Pure arithmetic only. No model, audio device, process or owner implementation.
#ifndef SHERPA_ONNX_CSRC_F104_DIAGNOSTIC_VALIDATION_H_
#define SHERPA_ONNX_CSRC_F104_DIAGNOSTIC_VALIDATION_H_
#include <array>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include "sherpa-onnx/csrc/f104-diarization-guard.h"

namespace sherpa_onnx { namespace f104 {
using LocalKey = std::pair<int32_t, int32_t>;
using ActivityKey = std::tuple<int32_t,int32_t,int32_t,int64_t,int64_t,int64_t,int64_t>;
using TurnKey = std::tuple<int32_t,int64_t,int64_t,int64_t,int64_t>;
inline ActivityKey ActivityValue(const Activity&a) {
  Require(!a.span.empty,"EMPTY_DIAGNOSTIC_SPAN");
  return {a.chunk,a.slot,a.cluster,a.span.start2,a.span.end2,a.span.start_ms,a.span.end_ms};
}
inline TurnKey TurnValue(const Turn&t) {
  Require(!t.span.empty,"EMPTY_DIAGNOSTIC_SPAN");
  return {t.cluster,t.span.start2,t.span.end2,t.span.start_ms,t.span.end_ms};
}
struct EvidencePopulation {
  std::vector<std::array<uint8_t,589>> masks;
  std::vector<Activity> activity;
  std::set<LocalKey> active;
};
constexpr int64_t kMinimumIsolatedFrames=10, kMinimumFrontendSamples=400;
inline EvidencePopulation ScorePopulation(const Diagnostics&d) {
  Require(d.samples>0 && d.samples<=kSamples && d.scores.size()<=size_t(kChunks),
          "POPULATION_LIMIT: diagnostics");
  if(d.status!="FAILURE")
    Require(d.scores.size()==size_t(Chunks(d.samples,160000,16000)),
            "ACTUAL_SEGMENTATION_EVIDENCE_REQUIRED");
  EvidencePopulation out;
  out.masks.resize(d.scores.size());
  constexpr uint8_t powerset[7]={0,1,2,4,3,5,6};
  uint64_t bytes=0;
  for(size_t chunk=0;chunk<d.scores.size();++chunk) {
    const auto&s=d.scores[chunk];
    Require(s.chunk==int64_t(chunk) && s.offset==int64_t(chunk)*16000 &&
            s.frames==589 && s.classes==7 && s.values.size()==589*7,
            "MALFORMED_SCORE_DESCRIPTOR");
    bytes=Add(bytes,589*7*4,kDiagnosticBytes);
    int64_t local=std::min<int64_t>(160000,d.samples-int64_t(chunk)*16000);
    Require(local>0,"MALFORMED_SCORE_DESCRIPTOR");
    int64_t keep=KeepFrames(589,local,991,270);
    for(int64_t frame=0;frame<589;++frame) {
      int cls=0;
      for(int c=0;c<7;++c) {
        Require(std::isfinite(s.values[frame*7+c]),"NONFINITE_SEGMENTATION_SCORE");
        // The existing Eigen maxCoeff selects the first equal maximum.
        if(s.values[frame*7+c]>s.values[frame*7+cls])cls=c;
      }
      out.masks[chunk][frame]=frame<keep ? powerset[cls] : 0;
    }
    for(int slot=0;slot<3;++slot) {
      int64_t start=-1;
      for(int64_t frame=0;frame<=keep;++frame) {
        bool on=frame<keep && (out.masks[chunk][frame]&(1<<slot));
        if(on && start<0){start=frame;out.active.insert({chunk,slot});}
        else if(!on && start>=0) {
          auto span=FrameSpan(start,frame,keep,local,991,270);
          span=Clip2(span.start2+2*int64_t(chunk)*16000,
                     span.end2+2*int64_t(chunk)*16000,d.samples);
          Require(out.activity.size()<size_t(kTurns),"POPULATION_LIMIT: diagnostic activity");
          out.activity.push_back({int32_t(chunk),slot,-1,span});start=-1;
        }
      }
    }
  }
  Require(bytes==d.score_bytes,"SCORE_BYTE_COUNT_MISMATCH");
  return out;
}
inline std::set<LocalKey> EligibleEmbeddings(const EvidencePopulation&e,int64_t samples) {
  std::set<LocalKey> eligible;
  for(size_t chunk=0;chunk<e.masks.size();++chunk) {
    int64_t local=std::min<int64_t>(160000,samples-int64_t(chunk)*16000);
    Require(local>0,"MALFORMED_SCORE_DESCRIPTOR");
    int64_t keep=KeepFrames(589,local,991,270);
    for(int slot=0;slot<3;++slot) {
      int64_t isolated=0,collected=0,start=-1;
      for(int64_t frame=0;frame<=keep;++frame) {
        uint8_t bits=frame<keep ? e.masks[chunk][frame] : 0;
        bool one_speaker=bits && !(bits&(bits-1));
        bool on=frame<keep && one_speaker && (bits&(1<<slot));
        if(on) {
          ++isolated;
          if(start<0)start=frame;
        } else if(start>=0) {
          auto span=FrameSpan(start,frame,keep,local,991,270);
          int64_t a=int64_t(chunk)*16000+span.start2/2;
          int64_t b=int64_t(chunk)*16000+(span.end2+1)/2;
          a=std::clamp<int64_t>(a,0,samples);b=std::clamp<int64_t>(b,0,samples);
          if(a<b)collected+=b-a;
          start=-1;
        }
      }
      if(isolated>=kMinimumIsolatedFrames && collected>=kMinimumFrontendSamples)
        eligible.insert({int32_t(chunk),slot});
    }
  }
  return eligible;
}
inline std::vector<Turn> EvidenceTurns(const EvidencePopulation&e,int64_t n,
                                     const std::map<LocalKey,int32_t>&mapping) {
  if(e.masks.empty())return {};
  int columns=0;
  for(const auto&m:mapping)if(m.second>=0)columns=std::max(columns,m.second+1);
  if(columns==0)return {};
  int64_t available=GlobalFrames(e.masks.size(),160000,16000,270);
  int64_t frames=KeepFrames(available,n,991,270);
  MatrixBytes(frames,columns,sizeof(uint16_t));
  std::vector<uint16_t> counts(frames*columns,0),weights(frames,0),totals(frames,0);
  for(size_t chunk=0;chunk<e.masks.size();++chunk) {
    int64_t offset=FrameOffset(chunk,16000,270);
    for(int local=0;local<589 && offset+local<frames;++local) {
      auto bits=e.masks[chunk][local];
      auto frame=offset+local;
      ++weights[frame];
      uint16_t known=0;
      for(int slot=0;slot<3;++slot)if(bits&(1<<slot)) {
        ++totals[frame];
        auto identity=mapping.at({chunk,slot});
        if(identity>=0)known|=uint16_t(1U<<identity);
      }
      for(int identity=0;identity<columns;++identity)if(known&(1U<<identity))
        ++counts[frame*columns+identity];
    }
  }
  std::vector<uint8_t> selected(frames*columns,0);
  for(int64_t frame=0;frame<frames;++frame) {
    Require(weights[frame]>0,"GLOBAL_RF_COVERAGE_REQUIRED");
    // The inherited average/round-half-up rule, using bounded integer counts.
    int k=std::min<int>((2*totals[frame]+weights[frame])/(2*weights[frame]),columns);
    std::vector<int> identities(columns);std::iota(identities.begin(),identities.end(),0);
    // Identical to the inherited TopkIndex comparator and partial_sort, including
    // its unspecified tied-index ordering. No new identity/tie algorithm.
    std::partial_sort(identities.begin(),identities.begin()+k,identities.end(),
        [&](int a,int b){return counts[frame*columns+a]>counts[frame*columns+b];});
    for(int i=0;i<k;++i)if(counts[frame*columns+identities[i]]>0)
      selected[frame*columns+identities[i]]=1;
  }
  std::vector<Turn> turns;
  for(int identity=0;identity<columns;++identity) {
    int64_t start=-1;
    for(int64_t frame=0;frame<=frames;++frame) {
      bool on=frame<frames && selected[frame*columns+identity];
      if(on && start<0)start=frame;
      else if(!on && start>=0) {
        auto span=FrameSpan(start,frame,frames,n,991,270);
        if(!span.empty && span.start_ms<span.end_ms) {
          Require(turns.size()<size_t(kTurns),"POPULATION_LIMIT: turns");
          turns.push_back({identity,span});
        }
        start=-1;
      }
    }
  }
  return turns;
}
inline void ValidateDiagnostics(const Diagnostics&d) {
  Require(d.schema=="f104.diarization.diagnostics/2.1" && d.state=="DEVELOPMENT" &&
      d.score_domain=="FINITE_NEURAL_SCORES_UNVERIFIED_DOMAIN","UNREVIEWED_NATIVE_CONTRACT");
  Require(d.mappings.size()<=size_t(kRows) && d.assignments.size()<=size_t(kRows) &&
      d.embedding_rejections.size()<=size_t(kRows) && d.activity.size()<=size_t(kTurns) &&
      d.turns.size()<=size_t(kTurns),"POPULATION_LIMIT: diagnostics");
  Require(d.status=="OK" || d.status=="SILENCE" || d.status=="PARTIAL_UNKNOWN" ||
      d.status=="UNIDENTIFIABLE_ACTIVITY" || d.status=="FAILURE","INVALID_DIAGNOSTIC_STATUS");
  Require((d.status=="FAILURE")==!d.error_code.empty(),"FAILURE_CODE_REQUIRED");
  Require(d.error_code.empty() || d.error_code=="POPULATION_LIMIT" ||
      d.error_code=="MALFORMED_INPUT" || d.error_code=="MODEL_ERROR","ERROR_CODE_INVALID");
  auto evidence=ScorePopulation(d);
  auto eligible=EligibleEmbeddings(evidence,d.samples);
  std::map<LocalKey,int32_t> mapping;
  bool unknown=false,known=false;
  for(const auto&m:d.mappings) {
    Require(m.chunk>=0 && m.chunk<int32_t(d.scores.size()) && m.slot>=0 && m.slot<3 &&
        m.cluster>=-1 && m.cluster<kClusters,"MALFORMED_LOCAL_MAPPING");
    Require(mapping.emplace(LocalKey{m.chunk,m.slot},m.cluster).second,"DUPLICATE_LOCAL_MAPPING");
    Require(m.reason==(m.cluster<0 ? "NO_VALID_ISOLATED_EMBEDDING" : "MAPPED"),"MAPPING_REASON_MISMATCH");
    Require(m.cluster<0 || eligible.count({m.chunk,m.slot}),
            "KNOWN_MAPPING_WITHOUT_ELIGIBLE_EMBEDDING");
    unknown|=m.cluster<0;known|=m.cluster>=0;
  }
  std::set<LocalKey> supplied;
  for(const auto&m:mapping)supplied.insert(m.first);
  Require(d.status!="SILENCE" || evidence.active.empty(),"FALSE_SILENCE");
  Require(supplied==evidence.active,"ACTIVE_MAPPING_POPULATION_MISMATCH");
  std::set<LocalKey> rejected;
  for(const auto&r:d.embedding_rejections) {
    LocalKey key{r.chunk,r.slot};
    Require(mapping.count(key) && mapping.at(key)==-1,"REJECTION_MAPPING_MISMATCH");
    Require(rejected.insert(key).second,"DUPLICATE_EMBEDDING_REJECTION");
    Require(r.reason=="MALFORMED_EMBEDDING_SHAPE" || r.reason=="NONFINITE_EMBEDDING" ||
        r.reason=="ZERO_OR_DEGENERATE_EMBEDDING" || r.reason=="INSUFFICIENT_ISOLATED_SAMPLES",
        "REJECTION_REASON_INVALID");
  }
  std::set<LocalKey> assignments,expected_assignments;
  for(const auto&m:mapping)if(m.second>=0)expected_assignments.insert(m.first);
  for(const auto&a:d.assignments) {
    LocalKey key{a.chunk,a.slot};
    Require(mapping.count(key) && mapping.at(key)>=0 && mapping.at(key)==a.cluster &&
        assignments.insert(key).second,"ASSIGNMENT_MAPPING_MISMATCH");
    if(a.has_assigned_distance)Require(std::isfinite(a.assigned_cosine_distance) &&
        a.assigned_cosine_distance>=0 && a.assigned_cosine_distance<=2,"INVALID_COSINE_DISTANCE");
    if(a.has_margin)Require(a.has_assigned_distance && std::isfinite(a.alternative_cosine_distance) &&
        a.alternative_cosine_distance>=0 && a.alternative_cosine_distance<=2 &&
        std::isfinite(a.margin) && a.margin>=-2 && a.margin<=2 &&
        std::abs(a.margin-(a.alternative_cosine_distance-a.assigned_cosine_distance))<=1e-12,
        "INVALID_ASSIGNMENT_MARGIN");
  }
  Require(assignments==expected_assignments,"ASSIGNMENTS_REQUIRED");
  std::vector<ActivityKey> actual_activity,expected_activity;
  for(const auto&a:evidence.activity) {
    auto copy=a;copy.cluster=mapping.at({a.chunk,a.slot});
    expected_activity.push_back(ActivityValue(copy));
  }
  for(const auto&a:d.activity)actual_activity.push_back(ActivityValue(a));
  std::sort(actual_activity.begin(),actual_activity.end());
  std::sort(expected_activity.begin(),expected_activity.end());
  Require(actual_activity==expected_activity,"ACTIVITY_POPULATION_MISMATCH");
  auto expected_turns=EvidenceTurns(evidence,d.samples,mapping);
  std::vector<TurnKey> actual_turns,turns;
  for(const auto&t:d.turns)actual_turns.push_back(TurnValue(t));
  for(const auto&t:expected_turns)turns.push_back(TurnValue(t));
  std::sort(actual_turns.begin(),actual_turns.end());std::sort(turns.begin(),turns.end());
  Require(actual_turns==turns,"TURN_POPULATION_MISMATCH");
  if(d.status!="FAILURE") {
    const char *expected=evidence.active.empty() ? "SILENCE" : !known ? "UNIDENTIFIABLE_ACTIVITY" :
        unknown ? "PARTIAL_UNKNOWN" : "OK";
    Require(d.status==expected,"STATUS_ACTIVITY_MISMATCH");
  }
}
}}  // namespace sherpa_onnx::f104
#endif
