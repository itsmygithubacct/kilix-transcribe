# Installed model descriptors

Serving a model installed through Kilix Content, and shared execution coordination. Moved from the README.

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

## Shared execution coordination

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
