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
authority or license receipt. Model/source selection for release, F100/F106
binding, resource admission, GPU profiles, corpus accuracy comparisons,
streaming recognition, microphone recording, VAD, diarization and the required
soak remain open. Unsupported diarization refuses explicitly. Commands that
need an unstaged runtime retain `RUNTIME_UNSELECTED` and exit 69.

The existing design ledger in `design/transcribe-candidate-v1.json` preserves
48 requirements, eight accepted source candidates and four unselected release
routes. Its unselected state describes qualification, not the presence of the
new development CPU path. No model weights, runtime executables, audio,
license receipts or release pins are included.

Run `make check` for the design/interface controls and unit tests. Process
supervision tests use explicit fake tools; passing them is not model-quality
or hardware-profile acceptance.
