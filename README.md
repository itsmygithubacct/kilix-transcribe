# kilix-transcribe

This repository contains the PREP7 design candidate and a CONT2 engine-neutral
implementation surface for the reusable local speech-to-text provider planned
by Plebian OS / Kilix 0.2.1.

It is not yet an engine-backed provider. The accepted F104 P0 source population is
represented exactly, while 0/4 engine routes, 0/8 source objects, and 0/1
release profiles are selected. Selection remains a P2 result after the formal
P1 contracts exist.

The planned command population is 7/7:

- `kilix-transcribe record`
- `kilix-transcribe file`
- `kilix-transcribe serve`
- `kilix-transcribe models`
- `kilix-transcribe status`
- `kilix-transcribe cancel`
- `kilix-transcribe unload`

The output-mechanics population is 4/4: plain text, JSON, WebVTT, and SRT. VAD,
language identification, word timestamps, optional supported translation, and
explicit-session diarization are retained requirements rather than silently
deferred features.

The pure-Python candidate under `src/kilix_transcribe/` implements only the
pieces that do not need a selected engine or frozen shared contract:

- bounded word and segment values with unstable-to-stable replacement rules;
- deterministic final-result serialization to all 4/4 output formats;
- a 6/6-state job lifecycle that commits at most 1/1 terminal result; and
- bounded candidate JSON framing, exact request types, kernel peer-UID checks,
  and 1/1 `SCM_RIGHTS` audio descriptor transport over Unix seqpacket sockets;
- private atomic transcript commits beneath provider-owned result names; and
- static `models` and `status` inspection with 0/4 routes, 0/8 source objects,
  and 0/1 release profiles selected.

All 5/5 operational commands fail closed with exit status 69 and the exact
provider refusal:

```text
KILIX_TRANSCRIBE_REFUSAL [RUNTIME_UNSELECTED] no transcription runtime or release profile is selected
```

Run the complete candidate check with:

```sh
make check
```

The check validates the exact values of the 48/48 requirement ledger and 8/8
accepted P0 source objects, 4/4 unselected architecture routes, 7/7 commands,
4/4 outputs, 11/11 committed checker negative controls, and 58/58 discovered
unit tests
using only Python's standard library.

The executable design ledger is
[`design/transcribe-candidate-v1.json`](design/transcribe-candidate-v1.json).
The implementation surface is explicitly a candidate internal API, not the P1
wire/catalog freeze. No engine source, model, audio, runtime, F100 record, F106
profile, package, remote, or release pin is included.
