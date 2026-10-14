# Source and modification notices

New F104 source-fix1 source is licensed under Apache-2.0; see sherpa-LICENSE
for the full licence text. This derivative has not been published or installed.

F104 source-fix2 modifications, 2026-10-09, also use Apache-2.0. The native
ProcessDetailed return paths and FD serializer now validate complete bounded
score/activity/turn populations; FrameSpan bounds samples before multiplication.
New pure validation modules and controls, bounded Python JSON numeric parsing,
range-first distance/schema checks and immutable ASR time bounds are included.
Existing imported source attribution and all four licence texts are preserved.

Imported sherpa-onnx source: Xiaomi Corporation and other retained authors,
v1.13.6 commit1cb484af5e69d3c7803c1eb0b3b5ab8041e0e911, Apache-2.0.
Each modified imported file carries an F104 source-fix1 modification notice.
Changes: English frontend settings/feature validation; two scoped CPU model
session options; strict segmentation geometry; one-chunk/empty-embedding guards;
finite normalization and checked clustering allocation/population; detailed
raw segmentation/mapping/assignment/activity; sample-bounded timing/turns.
The common session helper and imported architecture remain unchanged.

Imported CAMPPlus architecture: ModelScope/3D-Speaker commit
065629c313eaf1a01c65c640c46d77e61e9607b4, original headers retained,
Apache-2.0 text in3d-speaker-LICENSE. No checkpoint is conveyed. Exact English
weight eligibility remains the prior root determination; no new licence
acceptance or derived graph receipt is claimed.

Reference-only Torchaudio Kaldi source: pytorch/audio v2.7.0 commit
654fee8fd17784271be1637eac1293fd834b4e9a, full source/licence in parent
inputs/reference/torchaudio, BSD-2-Clause text in torchaudio-reference-LICENSE.
The reference adapter removes only ML imports and injects a future managed Torch
module; function bodies are unchanged. This is a recorded import modification,
not an installed Torchaudio package or executed parity result. The synthetic
scalar companion is an arithmetic control, not that primary implementation.

Kaldi-native-fbank reference source1.22.3 atb09e686fe2084732ddd30d1ef80acfc0f13eaf01
retains Xiaomi/Kaldi attribution and Apache-2.0 text in kaldi-native-fbank-LICENSE.
The captured reference headers/Compute implementation are unmodified and are
not an acquired or built binary dependency. Future conveyed binaries/wheels,
vendored libraries and derived graphs require their actual reviewed notices
and managed receipt/carrier populations, separately from these source texts.
