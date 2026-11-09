# kilix-transcribe

A local, reusable speech-to-text provider for Kilix. The current development
runtime executes a pinned whisper.cpp CPU engine behind a private Unix socket.
It accepts WAV, FLAC, and Ogg input, identifies language, supports transcription
and English translation, and returns timed segments and words as text, JSON,
WebVTT, or SRT. Voicebox consumes its transcript result descriptor without
retaining a second copy.

**Status:** a working development runtime and requirement ledger, not a
qualified release profile (see [Candidate boundaries](#candidate-boundaries)).
It is not installed with Plebian-OS by default: Kilix stages a pinned commit
with `kilix transcribe prepare` and runs it with `kilix transcribe serve`. Its
catalog model, `whisper-tiny-ggml`, is a Kilix Content asset that downloads on
first use after its licence screen; no weights ship in the ISO. Plebian-OS
provisioning separately installs the diarization conversion software (no
models) from this repository's `rc6-diarization-converter-20261014` release
asset; that code lives on the `rc6-release-20261014` branch, not on `main`.

## What it does

- `file`: transcribe or translate one audio file and print text, JSON, WebVTT
  or SRT.
- `record`: record one bounded microphone clip through Kilix Voice's managed
  microphone and transcribe it.
- `serve`: run the private service that owns the engine; `models`, `status`,
  `cancel` and `unload` query and control it.
- Reads models installed through Kilix Content, with a licence receipt check
  for every job; it never downloads code or models.

## Quick start

Stage the reviewed engine, decoder and model, then start the service and submit
a file from another shell. Every option and the build requirements for
`whisper-cli` are in [docs/running.md](docs/running.md).

```sh
python3 tools/stage_runtime.py \
  --destination "$XDG_DATA_HOME/kilix-transcribe/runtime" \
  --engine /path/to/whisper-cli --engine-sha256 <sha256> \
  --decoder /path/to/ffmpeg --decoder-sha256 <sha256> \
  --model /path/to/ggml-model.bin --model-sha256 <sha256> \
  --model-id whisper-tiny --model-revision <exact-revision>
PYTHONPATH=src python3 -m kilix_transcribe serve \
  --runtime-root "$XDG_DATA_HOME/kilix-transcribe/runtime"
PYTHONPATH=src python3 -m kilix_transcribe file input.wav --format srt
```

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

## Usage

| Command | What it does |
|---|---|
| `kilix-transcribe serve --runtime-root DIR` | Start the service for a staged runtime. |
| `kilix-transcribe serve ... --installed-asset whisper-tiny-ggml --content-root DIR --model-snapshot-bytes N` | Serve the Kilix Content model. |
| `kilix-transcribe file AUDIO [--format text\|json\|webvtt\|srt] [--task transcribe\|translate] [--language L]` | Transcribe one file. |
| `kilix-transcribe record --seconds N [--vad] [--device D]` | Record 1 to 120 seconds and transcribe it. |
| `kilix-transcribe models` / `status` | What the service has and is doing. |
| `kilix-transcribe cancel JOB` / `unload` | Cancel a job; confirm no model is held. |
| `make check` | Design ledger, command surface and unit tests. |

Inside Kilix the same provider is reached as `kilix transcribe prepare`,
`serve`, `status`, `models`, `file`, `record`, `unload` and `cancel`. Kilix's
other `kilix transcribe` verbs (`engines`, `install`, `run`, `chunk`, `probe`)
belong to Kilix's own file-transcription front end, not to this repository.

## How it works

```
kilix-transcribe file / record      Voicebox
          \                            /
     same-UID Unix socket kilix-transcribe.sock, bounded JSON
                         |
            kilix-transcribe serve   (one job, bounded connections)
                         |
          per-job Linux descendant supervisor
              |                     |
        ffmpeg decoder        whisper-cli engine   <- sealed memfd copies
                         |
     result JSON through one read-only descriptor (<= 16 MiB)
```

Each job has a dedicated Linux descendant supervisor. A deadline, client
disconnect, service shutdown, or cancellation stops and reaps the worker and
decoder/engine children, including children that create another session.
Inputs, engines and model bytes are copied into sealed memory files after size
and SHA-256 checks, so a replaced file on disk cannot substitute bytes. The
full description is in [docs/running.md](docs/running.md); model selection
through Kilix Content and shared device leases are in
[docs/installed-models.md](docs/installed-models.md).

## Candidate boundaries

This is a working development runtime, **not a qualified release profile**.
Its digest-bound `kilix.transcribe.runtime/v1` manifest is not an F100 install
authority or license receipt. Production asset admission, F106 binding,
resource admission, GPU profiles, combined corpus/soak qualification,
streaming recognition, neural VAD and diarization remain open. The managed
recording and optional legacy energy detector below are development paths.
Unsupported diarization refuses explicitly. Commands that
need an unstaged runtime retain `RUNTIME_UNSELECTED` and exit 69.

The existing design ledger in `design/transcribe-candidate-v1.json` preserves
48 requirements, eight accepted source candidates and four unselected release
routes. Its unselected state describes qualification, not the presence of the
new development CPU path. No model weights, runtime executables, audio,
license receipts or release pins are included.

## Documentation

- [docs/running.md](docs/running.md): staging a development runtime, the
  service, job supervision and descriptor bounds.
- [docs/installed-models.md](docs/installed-models.md): installed model
  descriptors, receipt checks, tests, and shared execution coordination.
- [docs/recording-and-client.md](docs/recording-and-client.md): managed
  microphone recording and controlled client delivery.
- [design/transcribe-candidate-v1.json](design/transcribe-candidate-v1.json):
  the requirement ledger.

## Related repositories

- [kilix](https://github.com/itsmygithubacct/kilix#readme): `kilix transcribe`,
  which stages and runs this provider.
- [kilix-content](https://github.com/itsmygithubacct/kilix-content#readme) and
  [kilix-license](https://github.com/itsmygithubacct/kilix-license#readme): the
  model catalog and the licence receipts every job checks.
- [kilix-voice](https://github.com/itsmygithubacct/kilix-voice#readme): the
  managed microphone and the voice energy detector `record` uses.
- [kilix-voicebox](https://github.com/itsmygithubacct/kilix-voicebox#readme):
  the studio candidate that consumes transcript descriptors.
- [kilix-whisper-stt](https://github.com/itsmygithubacct/kilix-whisper-stt#readme):
  the separate dictation provider.
- [kilix-qwen-tts](https://github.com/itsmygithubacct/kilix-qwen-tts#readme):
  the synthesis provider built the same way.

## Part of Plebian-OS

kilix-transcribe is one of the repositories behind
[Plebian-OS](https://github.com/itsmygithubacct/plebian-os#readme), a Debian
install whose desktop is [Kilix](https://github.com/itsmygithubacct/kilix#readme)
running the [Kilix 95](https://github.com/itsmygithubacct/kilix-95#readme)
desktop through [Pleb](https://github.com/itsmygithubacct/pleb#readme).
Kilix stages a pinned commit when a user prepares transcription, and Plebian-OS
installs its diarization converter release asset. The map of every repository
and how it ships is on the
[components page](https://github.com/itsmygithubacct/plebian-os/blob/main/docs/components.md).

## License

This wrapper source is licensed under MIT; see [LICENSE](LICENSE). Upstream
engines, dependencies and models retain their separate terms.
