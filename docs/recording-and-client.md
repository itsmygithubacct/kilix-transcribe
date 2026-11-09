# Managed recording and client delivery

Recording one bounded microphone clip through Kilix Voice, and cancellable client calls. Moved from the README.

## Managed microphone recording

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
