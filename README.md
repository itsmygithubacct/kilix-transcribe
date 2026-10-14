# kilix-transcribe

A local, reusable speech-to-text provider for Kilix. The current development
runtime executes a pinned whisper.cpp CPU engine behind a private Unix socket.
It accepts WAV, FLAC, and Ogg input, identifies language, supports transcription
and English translation, and returns timed segments and words as text, JSON,
WebVTT, or SRT. Voicebox consumes its transcript result descriptor without
retaining a second copy.

## Run an explicitly staged development runtime

The provider does not download code or models. Stage the three reviewed inputs
in a new private directory, using their independently verified SHA-256 values:

```sh
python3 tools/stage_runtime.py \
  --destination "$XDG_DATA_HOME/kilix-transcribe/runtime" \
  --engine /path/to/whisper-cli --engine-sha256 <sha256> \
  --decoder /path/to/ffmpeg --decoder-sha256 <sha256> \
  --model /path/to/ggml-model.bin --model-sha256 <sha256> \
  --model-id whisper-tiny --model-revision <exact-revision>
PYTHONPATH=src python3 -m kilix_transcribe serve \
  --runtime-root "$XDG_DATA_HOME/kilix-transcribe/runtime"
```

Build `whisper-cli` from whisper.cpp commit
`371b5a7561823ab2bb32142d2751e35e7534727b`, with CPU support and
`GGML_NATIVE=OFF`, `GGML_CUDA=OFF`, and `BUILD_SHARED_LIBS=OFF`. The staging
command binds supplied bytes; it cannot certify how an executable was built.
The decoder's shared-library closure remains an installation responsibility.
`XDG_RUNTIME_DIR` must name a private, owned directory. The service creates
`kilix-transcribe.sock` with mode 0600 and checks the connecting peer's UID.

From another process with the same runtime directory:

```sh
PYTHONPATH=src python3 -m kilix_transcribe file input.wav --format srt
PYTHONPATH=src python3 -m kilix_transcribe file input.flac --task translate
PYTHONPATH=src python3 -m kilix_transcribe models
PYTHONPATH=src python3 -m kilix_transcribe status
PYTHONPATH=src python3 -m kilix_transcribe cancel <job-id>
PYTHONPATH=src python3 -m kilix_transcribe unload
```

Each job has a dedicated Linux descendant supervisor. A deadline, client
disconnect, service shutdown, or cancellation stops and reaps the worker and
decoder/engine children, including children that create another session. Temporary audio and transcripts are removed by the supervisor.
Each job loads its model; `unload` confirms there is no persistent model and
refuses while a worker is active. There is one active job and a bounded number
of connections. Additional jobs receive `BUSY`.

Inputs are copied from one read-only descriptor after size and SHA-256 checks.
Jobs cannot supply paths, executable names, model locations, or URLs. Engines
and model bytes are copied into verified, sealed memory files and executed/read
through inherited descriptors; a later installation-path replacement cannot
substitute bytes. Runtime ancestor identities are checked without following
symlinks. Result
JSON travels through one read-only descriptor with size and SHA-256 metadata,
up to 16 MiB, while the control message limit remains 64 KiB. The service and
worker do not log input audio or transcripts.

## Candidate boundaries

This is a working development runtime, **not a qualified release profile**.
Its digest-bound `kilix.transcribe.runtime/v1` manifest is not an F100 install
authority or license receipt. Production asset admission, F106 binding,
resource admission, GPU profiles and combined corpus/soak qualification
remain open. Incremental recognition has a development implementation;
physical/model qualification and neural VAD remain open. The managed
recording and optional legacy energy detector below are development paths.
Speaker turns require the separate exact runtime and an explicit session. Commands that
need an unstaged runtime retain `RUNTIME_UNSELECTED` and exit 69.

The CPU speaker-turn route uses pinned pyannote 3.0 segmentation and a locally
exported English CAM++ graph. Select the receipt-covered `kilix-diarization-cpu`
asset with `serve --diarization-runtime PATH --diarization-asset
kilix-diarization-cpu`, alongside the installed Tiny selection. Each request
must use `file INPUT --task diarize --diarization-session --format json`.
Speaker turns form a separate annotation bound to the unchanged recognized
words/timestamps. Anonymous labels are new for every request; no embedding or
speaker identity is retained. Unidentifiable activity remains explicit.
Streaming diarization is refused; transcript streaming remains available for
ordinary transcription/translation. These checks do not claim a qualified
speaker-accuracy or sustained-load profile.

`tools/build_diarization_native.py` builds the CPU runner from exact local
software archives without fetching. `tools/stage_diarization_tool.py` creates
the optional receipt-gated conversion tool from committed product source and
the pinned Python/export environment, and conveys the exact native-build Eigen
source archive with its MPL notice (`--eigen-source`). Conversion reads only the approved
English checkpoint with `weights_only=True` inside a network-denied, bounded
CPU scope and requires the reproducible selected ONNX digest. Installed
runtime inference loads ONNX only. `tools/stage_diarization_runtime.py` also
supports explicit local development staging; it never creates a model receipt.

The existing design ledger in `design/transcribe-candidate-v1.json` preserves
48 requirements, eight accepted source candidates and four unselected release
routes. Its unselected state describes qualification, not the presence of the
new development CPU path. No model weights, runtime executables, audio,
license receipts or release pins are included.

## Installed model descriptors

With the reviewed `kilix-content` installed-asset API provisioned, select the
catalog model at service startup:

```sh
kilix-transcribe serve --runtime-root /absolute/runtime \
  --installed-asset whisper-tiny-ggml \
  --content-root /absolute/installed-content \
  --model-snapshot-bytes 80000000
```

The explicit byte ceiling bounds model snapshots; it is not hardware admission.
The runtime manifest still names the exact engine, decoder and model digests.
Engine/decoder files remain under `runtime-root`; `model.bin` comes only from
the authorized installed population. The catalog's provider, consumer version,
model revision and full model population must match. Missing assets or receipts
refuse without falling back to pathname models.

To stage the tools and manifest without copying an installed model, use the
same Content selection with `tools/stage_runtime.py`. Supply the engine and
decoder paths/digests, model digest, model ID and revision shown in the
development staging example above, and replace `--model` with:

```sh
--installed-asset whisper-tiny-ggml \
--content-root /absolute/installed-content \
--model-snapshot-bytes 80000000
```

Staging verifies receipt coverage and the installed population before
publishing the destination. Installed service startup repeats those checks
before creating its socket; missing consent or damaged bytes cannot report
readiness. The runtime directory contains only the tools and manifest.

The provider uses the selected `kilix-content` packaged catalog and
`kilix-license` shared receipt store. Receipt coverage is checked again for
every job; the store retains no open descriptor. Each job checks the complete
installed file population under its cancellation/deadline checks and an
additional 120-second snapshot ceiling, then copies verified bytes into sealed
read-only memory files.
The provider requires read-only sealed model descriptors and independently
hashes the actual bytes before passing them to the owned worker. No model path,
catalog or release identity is accepted on wire, and the provider creates no
license receipt. This supplies packaged receipt binding, not full transaction
or resource-profile qualification.

The installed integration tests use explicit synthetic packaged catalogs with
receipts captured through the real agreement authority in a private store.
The tests exercise the selected Content layout, licence coverage and production
snapshot implementation without granting any model licence. Put the selected
`kilix-content` and `kilix-license` packages on `PYTHONPATH` to run them; they
report skips when those optional packages are absent. Missing or foreign
receipts, revoked coverage, changed bytes, undeclared members, symlinks,
cancellation and malformed descriptors refuse without starting inference.

Run `make check` for the design/interface controls and unit tests. Process
supervision tests use explicit fake tools; passing them is not model-quality
or hardware-profile acceptance.


### Shared execution coordination

`serve --lease-device cpu-development` explicitly selects the optional
`kilix_device_lease` v1 API from the `kilix-device-lease` distribution in
`kilix-system-monitor`. Install that distribution into the provider's
interpreter before selecting the policy. The same private default namespace coordinates
all participating providers; `--lease-namespace` selects a shared private
absolute namespace for isolated deployments or tests. A namespace requires a
device label. Selecting this policy requires the API to be installed and never
falls back to uncoordinated execution. The label grants coordination only;
it does not select a GPU, establish measured fit, or qualify a resource profile.
The explicitly selected CPU runtime continues to execute CPU inference.

A job waits for the shared grant before copying or allocating model bytes. Its
normal provider job slot remains occupied while queued; cancellation,
disconnection and the original deadline remain effective, with bounded queued
progress events. The grant is inherited by the dedicated supervisor and retained
through descendant teardown. Transcription also propagates it through the
decoder and inference worker; synthesis retains another copy in the namespace
launcher for the sandbox lifetime.

A private seqpacket channel carries exactly one cleanup acknowledgment from the
supervisor after all its owned descendants are reaped. The provider checks the
kernel sender PID/UID/GID and complete message before acknowledging the grant.
Normal success, engine errors and cancellation can release only after that
proof. A spawn that created no child can release directly. Missing, malformed
or wrong-sender proof makes the service unavailable and leaves shared ownership
quarantined, even after all guard descriptors disappear. An unload or service
restart does not reset persistent quarantine. There is no automatic recovery
from unproven ownership in this API. Unrelated embedding-process children are
never adopted or reaped by the provider.

Existing direct development execution remains available without this explicit
policy, and still requires supervisor cleanup proof. Its in-process unavailable
state cannot establish persistent coordination across provider restarts. Shared
lease, installed asset, hardware admission, microphone policy and release
qualification are separate requirements; passing local controls supplies none
of the unmeasured qualifications.


### Installing the Python provider

The source builds a standard Python wheel for the selected Python 3.12 runtime,
with the `kilix-transcribe` console entry point and the bounded Python client.
The wheel contains the provider package only. Its build tools are pinned to
setuptools 78.1.0 and wheel 0.45.1; the provider itself uses the standard library.
Engine executables, model bytes, optional `kilix-content` authority and shared
`kilix-voice` coordination are installed and selected separately. Building a
wheel supplies no model download. This wrapper source is licensed under MIT;
see [LICENSE](LICENSE). Upstream engines, dependencies and models retain
their separate terms.

### Managed microphone recording

With the managed `kilix-voice` microphone API installed and the transcription
service ready, explicitly record one bounded clip:

```sh
kilix-transcribe record --seconds 30 --format text
kilix-transcribe record --seconds 60 --vad --task translate --format srt
```

`--seconds` selects 1 to 120 seconds; `--timeout` covers recording and recognition
and must allow additional recognition time. `--device` selects a validated
PulseAudio source token. `--vad` uses the existing voice energy detector to end
after speech and trailing silence. It keeps the original captured samples; it
does not select Silero or establish recognition/silence quality. A bare `record`
retains the unselected introspection surface and opens no microphone.

The CLI prints microphone ownership phases to stderr. Recording is indicated
before the helper opens audio; `inactive` requires proved descendant cleanup.
An unresolved helper/supervisor remains `unavailable` and its shared microphone
grant stays quarantined. Ctrl-C, termination, a departed output receiver,
provider failure, unknown/locked logind state or the absolute capture limit
stop recording. The normal duration limit submits the clip only after cleanup;
cancellation, lost frames, helper errors and unproved cleanup never submit it.

Capture uses the same per-user microphone namespace as other cooperating voice
clients, and keeps bounded mono16k PCM in memory. Recognition begins after all
recorder descendants are proved reaped. Its read-only WAV descriptor goes
through the existing provider byte checks. No raw recording or transcript is
persisted by default, and the CLI never types keys or Enter. Cancel during
recognition requests cancellation for the exact job ID, closes the client's
channel and raises CANCELED without returning a transcript. Local cancellation
or an ACK does not prove provider cleanup; busy/unavailable status remains
authoritative before a successor request.
The same actual lock-state authority is retained through recognition and final
delivery; lock loss or an unknown state refuses a late transcript as well.

Embedders use `recording.capture_wav` with a required visible `indicator`
callback and bounded cancellation/disconnect/lock authorities, or
`recording.record_and_transcribe` for the combined flow. Recorder command
configuration and private test namespaces are local embedding controls, never
accepted from provider IPC. The optional Python API must be installed; missing
managed microphone support refuses instead of opening an unmanaged device.
Synthetic recorder/ASR tests exercise these boundaries without physical capture.

## Controlled client delivery

The Python client accepts `client_request(..., cancelled=callback)` for submit
operations. The callback must promptly return a boolean. Controlled calls keep
the requested deadline and observe cancellation during receive waits. On
cancellation the client attempts a short cancel request for that submitted job,
closes its own channel and descriptors, and raises `CANCELED`. This is local
cancellation: neither it nor a cancel ACK proves provider cleanup. Check provider
status before a successor job; busy or unavailable remains authoritative. No
background request thread is retained. Borrowed input descriptors stay open.
The client rechecks the original deadline after validation, immediately before
returning output. Expired delivery is refused with `DEADLINE_EXCEEDED`, including
short control responses; received descriptors are still closed on refusal.

Installed model directory chains must be owned by the current user or root
and must not permit group or other writes. The nominated Content root and all
held descendant directories are checked; replacing the root with a symlink
refuses. Member files still require current-user ownership and exact catalog
bytes. This directory policy does not establish hardware admission or release
qualification.

## Incremental transcript revisions

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
