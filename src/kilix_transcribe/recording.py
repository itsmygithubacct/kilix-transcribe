"""Explicit, bounded microphone capture; recognition starts after proved close."""
from __future__ import annotations

from dataclasses import dataclass
import io
import math
import os
from pathlib import Path
import select
import tempfile
import threading
import time
import uuid
import wave

from .protocol import ProtocolError

RATE = 16000
FRAME_MS = 20
MAX_SECONDS = 120


@dataclass(frozen=True)
class Recording:
    wav: bytes
    duration_ms: int
    ended_by: str
    speech_detected: bool | None


def _boolean(callback, default=False):
    if callback is None:
        return default
    try:
        value = callback()
    except Exception as error:
        raise ProtocolError("CANCELED", "capture control is unavailable") from error
    if type(value) is not bool:
        raise ProtocolError("CANCELED", "capture control is unknown")
    return value


def _seconds(value):
    # Range checks precede float conversion even for arbitrarily large ints.
    if type(value) not in (int, float) or not 1 <= value <= MAX_SECONDS or not math.isfinite(value):
        raise ProtocolError("INVALID_REQUEST", "capture seconds must be between 1 and 120")
    return float(value)


def capture_wav(*, seconds, indicator, cancelled=None, disconnected=None, locked=None,
                provider_alive=None, namespace=None, config=None, vad=False) -> Recording:
    """Return original captured mono16k PCM only after authenticated teardown.

    The caller must show every indicator phase, including persistent unavailable.
    Explicit configuration is an embedding API, never accepted from IPC input.
    Energy VAD optionally ends the recording after speech and trailing silence;
    it neither edits the captured samples nor selects an ASR/VAD model profile.
    """
    seconds = _seconds(seconds)
    if (not callable(indicator) or type(vad) is not bool
            or any(callback is not None and not callable(callback)
                   for callback in (cancelled, disconnected, locked, provider_alive))
            or config is not None and type(config) is not dict):
        raise ProtocolError("INVALID_REQUEST", "capture requires explicit bounded controls")
    try:
        from voicelib.audio import AudioError, MicCapture
        from voicelib.microphone import CaptureControl, CaptureError
        from voicelib.vad import Vad
    except ImportError as error:
        raise ProtocolError("UNSUPPORTED_CAPABILITY", "managed microphone support is not installed") from error

    def gone():
        return _boolean(disconnected) or not _boolean(provider_alive, default=True)

    if (config is not None and (set(config) - {"audio"}
            or type(config.get("audio", {})) is not dict
            or set(config.get("audio", {})) - {"device_in", "capture_cmd"})):
        raise ProtocolError("INVALID_REQUEST", "unsupported microphone configuration")
    options = {"audio": dict((config or {}).get("audio", {}))}
    options["audio"].update(rate=RATE, frame_ms=FRAME_MS)
    deadline = time.monotonic() + seconds
    control = CaptureControl(deadline=deadline, cancelled=cancelled, disconnected=gone,
                             locked=locked, indicator=indicator, namespace=namespace)
    capture = MicCapture(options, control=control)
    detector = Vad(options) if vad else None
    limit = int(seconds * RATE) * 2
    pcm = bytearray()
    ended = "duration"
    speech = False if vad else None
    try:
        control.check()
        capture.start()
        while time.monotonic() < deadline and len(pcm) < limit:
            control.check()
            frame = capture.read(timeout=min(0.05, max(0, deadline - time.monotonic())))
            if capture.phase == "unavailable":
                raise ProtocolError("SUPERVISOR_FAILED", "microphone cleanup remains unproven")
            if capture.error is not None:
                raise ProtocolError("PROVIDER_ERROR", "microphone capture failed")
            if capture.overruns:
                raise ProtocolError("LIMIT_EXCEEDED", "microphone frames were lost")
            if frame is None:
                if capture.phase == "inactive":
                    ended = "source-ended"
                    break
                continue
            if type(frame) is not bytes or len(frame) != RATE * FRAME_MS // 1000 * 2:
                raise ProtocolError("INVALID_RESPONSE", "microphone returned an invalid frame")
            pcm.extend(frame[:limit - len(pcm)])
            if detector is not None:
                state = detector.feed(frame)
                if state == "speech_start":
                    speech = True
                elif state == "speech_end":
                    ended = "silence"
                    break
    except CaptureError as error:
        if error.code != "deadline" or time.monotonic() < deadline:
            raise ProtocolError("CANCELED", "microphone control stopped capture") from error
    except (AudioError, RuntimeError) as error:
        raise ProtocolError("PROVIDER_ERROR", "microphone capture failed") from error
    finally:
        proven = capture.stop()
        if not proven or not capture.cleanup_complete:
            raise ProtocolError("SUPERVISOR_FAILED", "microphone cleanup remains unproven")
    # A normal duration limit is not permission to ignore a simultaneous stop,
    # screen lock, provider failure or departed consumer.
    if _boolean(cancelled) or gone() or _boolean(control.locked):
        raise ProtocolError("CANCELED", "capture consumer or lock state stopped the job")
    if capture.overruns or capture.error is not None:
        raise ProtocolError("PROVIDER_ERROR", "microphone capture was incomplete")
    if not pcm:
        raise ProtocolError("INVALID_REQUEST", "microphone captured no complete audio frame")
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        writer.writeframes(pcm)
    return Recording(output.getvalue(), len(pcm) * 1000 // (RATE * 2), ended, speech)


class ProviderLifetime:
    """Bounded ready-state observation while the microphone is open."""
    def __init__(self, directory):
        self.directory = directory
        self._lock = threading.Lock()
        self._next = 0.0
        self._alive = True

    def __call__(self):
        from .service import client_request, request_value
        with self._lock:
            if not self._alive or time.monotonic() < self._next:
                return self._alive
            try:
                value = client_request(self.directory, request_value("status", timeout=0.2))
                self._alive = value.get("provider_state") == "ready" and value.get("worker_active") is False
            except (ProtocolError, OSError, ValueError):
                self._alive = False
            self._next = time.monotonic() + 0.25
            return self._alive


def receiver_closed(descriptor):
    """Observe a CLI pipe/terminal closing without emitting text or keystrokes."""
    try:
        poller = select.poll()
        poller.register(descriptor, select.POLLERR | select.POLLHUP | select.POLLNVAL)
        return bool(poller.poll(0))
    except (OSError, ValueError):
        return True


def record_and_transcribe(directory: Path, *, seconds, indicator, task="transcribe", language=None,
                          output="text", timeout=300, cancelled=None, disconnected=None, locked=None,
                          namespace=None, config=None, vad=False):
    """One explicit capture and one bound provider job; no persistent audio."""
    from .service import client_request, request_value
    import hashlib
    seconds = _seconds(seconds)
    if type(timeout) not in (int, float) or not seconds < timeout <= 3600 or not math.isfinite(timeout):
        raise ProtocolError("INVALID_REQUEST", "timeout must include capture and recognition within 3600 seconds")
    try:
        from voicelib.microphone import LogindLockState
    except ImportError as error:
        raise ProtocolError("UNSUPPORTED_CAPABILITY", "managed microphone support is not installed") from error
    lock_state = LogindLockState() if locked is None else locked
    started = time.monotonic()
    # Validate the entire requested transcript selection before opening audio.
    arguments = {"task": task, "language": language, "output": output, "audio_fd": 0,
                 "audio": {"media_type": "audio/wav", "byte_length": 44, "sha256": "0" * 64}}
    request_value("submit", job_id="capture-preflight", args=arguments, timeout=timeout)
    models = client_request(directory, request_value("models", timeout=min(1, timeout)))
    rows = models.get("models")
    if type(rows) is not list or not any(type(row) is dict and type(row.get("capabilities")) is list
                                       and task in row["capabilities"] for row in rows):
        raise ProtocolError("UNSUPPORTED_CAPABILITY", "provider cannot perform the selected recording task")
    lifetime = ProviderLifetime(directory)
    if not lifetime():
        raise ProtocolError("BUSY", "transcription provider is not ready for recording")
    if timeout - (time.monotonic() - started) <= seconds:
        raise ProtocolError("DEADLINE_EXCEEDED", "insufficient deadline remains for recording")
    clip = capture_wav(seconds=seconds, indicator=indicator, cancelled=cancelled, disconnected=disconnected,
                       locked=lock_state, provider_alive=lifetime, namespace=namespace, config=config, vad=vad)
    arguments["audio"] = {"media_type": "audio/wav", "byte_length": len(clip.wav),
                          "sha256": hashlib.sha256(clip.wav).hexdigest()}
    remaining = timeout - (time.monotonic() - started)
    if remaining <= 0:
        raise ProtocolError("DEADLINE_EXCEEDED", "recording job deadline expired")
    value = request_value("submit", job_id=uuid.uuid4().hex, args=arguments, timeout=remaining)
    with tempfile.TemporaryFile() as writer:
        writer.write(clip.wav)
        writer.flush()
        descriptor = os.open(f"/proc/self/fd/{writer.fileno()}", os.O_RDONLY | os.O_CLOEXEC)
        try:
            return client_request(directory, value, descriptor,
                                  cancelled=lambda: (_boolean(cancelled) or _boolean(disconnected)
                                                     or _boolean(lock_state)))
        finally:
            os.close(descriptor)
