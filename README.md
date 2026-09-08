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
resource admission, GPU profiles, corpus accuracy comparisons,
streaming recognition, microphone recording, VAD, diarization and the required
soak remain open. Unsupported diarization refuses explicitly. Commands that
need an unstaged runtime retain `RUNTIME_UNSELECTED` and exit 69.

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

Receipt storage opens once during bounded service startup and closes after
shutdown. Each job rechecks durable receipts and all installed files under its
cancellation/deadline checks and an additional 120-second snapshot ceiling.
The provider requires read-only sealed model descriptors and independently
hashes the actual bytes before passing them to the owned worker. No model path,
catalog or release identity is accepted on wire, and the provider creates no
license receipt. This supplies packaged receipt binding, not full transaction
or resource-profile qualification.

The installed integration tests use explicit synthetic packaged catalogs with
real receipt/installer operations. Put the reviewed `kilix-content` package on
`PYTHONPATH` to run them; they report skips when that optional API is absent.

Run `make check` for the design/interface controls and unit tests. Process
supervision tests use explicit fake tools; passing them is not model-quality
or hardware-profile acceptance.


### Shared execution coordination

`serve --lease-device cpu-development` explicitly selects the optional
`voicelib.device_leases` v1 API. The same private default namespace coordinates
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
wheel supplies neither a model download nor a source-license grant.


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
