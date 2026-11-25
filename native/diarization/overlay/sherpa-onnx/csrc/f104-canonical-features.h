// F104 source-fix1, Apache-2.0. Canonical English settings, parity still pending.
#ifndef SHERPA_ONNX_CSRC_F104_CANONICAL_FEATURES_H_
#define SHERPA_ONNX_CSRC_F104_CANONICAL_FEATURES_H_
namespace sherpa_onnx { namespace f104 {
template<class Config> inline void EnglishFeatures(Config *c) {
  c->sampling_rate=16000;c->normalize_samples=true;c->feature_dim=80;
  c->f104_english_canonical=true;c->low_freq=20;c->high_freq=0;
  c->snip_edges=true;c->dither=0;c->frame_shift_ms=10;c->frame_length_ms=25;
  c->remove_dc_offset=true;c->preemph_coeff=0.97f;c->window_type="povey";
  c->round_to_power_of_two=true;c->is_librosa=false;
}
}} // namespace sherpa_onnx::f104
#endif
