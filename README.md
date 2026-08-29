# kilix-transcribe

This repository contains the PREP7 design candidate for the reusable local
speech-to-text provider planned by Plebian OS / Kilix 0.2.1.

It is a design and review surface, not a working provider. The accepted F104
P0 source population is represented exactly, while 0/4 engine routes and 0/8
source objects are selected for release use. Selection remains a P2 result
after the formal P1 contracts exist.

The planned command population is 7/7:

- `kilix-transcribe record`
- `kilix-transcribe file`
- `kilix-transcribe serve`
- `kilix-transcribe models`
- `kilix-transcribe status`
- `kilix-transcribe cancel`
- `kilix-transcribe unload`

The planned output population is 4/4: plain text, JSON, WebVTT, and SRT. VAD,
language identification, word timestamps, optional supported translation, and
explicit-session diarization are retained requirements rather than silently
deferred features.

Run the complete candidate check with:

```sh
make check
```

The check validates the exact values of the 48/48 requirement ledger and 8/8
accepted P0 source objects, 4/4 unselected architecture routes, 7/7 commands,
4/4 outputs, 4/4 committed negative controls, and 12/12 unit tests using only
Python's standard library.

The executable design ledger is
[`design/transcribe-candidate-v1.json`](design/transcribe-candidate-v1.json).
No source, model, audio, runtime, F100 record, F106 profile, package, remote, or
release pin is included.
