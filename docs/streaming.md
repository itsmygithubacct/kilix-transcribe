# Incremental transcript revisions

`file input.wav --stream` and `record --seconds 30 --stream` show provisional
segment revisions on stderr as `KILIX_TRANSCRIBE_UPDATE` followed by JSON.
Stdout contains the corrected final text/JSON/SRT/WebVTT only. A revision with
`stable:false` may change or disappear; consumers replace it by segment ID.
Failure or cancellation leaves those previews provisional. Recognition begins
after bounded decoding and, for recording, proved microphone close.

Python embedders first call `service.negotiate_stream(directory)`, then set
`args["stream"]="segments-v1"` in a submit request and pass
`client_request(..., on_update=callback)`. The callback receives a typed
`SegmentUpdate`, must return promptly, and may not treat an update as a final
commit. `record_and_transcribe(..., on_update=callback)` negotiates before
opening the microphone and provides the same revisions during recognition.
Older peers refuse streaming; calls without the feature keep their existing
final-only route. Old hello requests still report minor 0. Feature hello
requests select minor 1 and advertise the exact frame, update and population
limits. Diarization remains unsupported.

The pinned CLI's native new-segment callback supplies decoded text during
inference. Each 20-second interval first decodes a five-second owned prefix
with up to two seconds of left context, then its full interval with up to two
seconds of context on each side. Full-window JSON corrects and stabilizes that
interval. Complete textual units are reconstructed before midpoint ownership:
leading whitespace or a gap over 500 ms starts a unit, while continuation
pieces and punctuation stay attached. Runs without whitespace remain indivisible
units; this policy does not claim linguistic word segmentation. Exact raw text
and full word times survive ownership. Stable segment envelopes expand into
the existing context when necessary to contain a crossing word. Already
committed token occurrences are reconciled by exact raw text and absolute
timing before assembling a later window, so a visible continuation cannot
repeat an already committed unit. Matching words at different times remain
distinct; complete units have no artificial two-second length limit. IDs increase
per interval; revisions and update sequence numbers
are contiguous. A new core requires the previous core to be stable and cover
its complete ownership interval; updates and final duration must agree with
the core's time range. Advancing also rechecks any prior nominal-end EOF word
before delivering the next update. Empty stable revisions retract previews and produce no subtitle
cue. Streaming interval grouping can differ from full-file decoding. Corpus
accuracy at boundaries and actual first-text latency remain unqualified.

Control packets stay at most 64 KiB, text at most 32 KiB/update and words at
most 256/update. A job allows 180 intervals, 4,096 updates, 1 MiB aggregate
text and a 16 MiB final result, within the existing one-hour/512 MiB input
bounds. Unstable updates are coalesced at 20 Hz. Window JSON/stdout is bounded
to 1 MiB. There is no application update queue. An update receiver blocked for
200 ms fails with `SLOW_CONSUMER`; a disconnected receiver may observe EOF.
CLI stderr diagnostics, including refusal and argument errors, use bounded
best-effort delivery too; a blocked or closed receiver can lose diagnostics.
A separate nonblocking open-file description leaves inherited stderr flags
unchanged and retries temporary pipe/terminal backpressure within the original
200 ms budget. Diagnostic delivery spawns no child; stopping or losing the
caller cannot leave an output helper behind. Unsupported stderr targets refuse
required previews and may lose best-effort messages. Output to regular files
remains subject to filesystem scheduling and is not a hard real-time guarantee.
The owned supervisor reaps decoder, inference and escaped descendants before
the shared grant is released or a terminal result is sent. Source tests and a
weight-free pinned build establish this route; licensed real-model, fitted H1,
live microphone and installed acceptance are still required.
