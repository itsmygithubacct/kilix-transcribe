"""Local transcription client and installed-runtime service entry point."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import time
import sys
import uuid
from collections.abc import Sequence

from .surface import COMMANDS, RuntimeUnselected, SurfaceError, inspect_command
from .protocol import ProtocolError


def _stderr_write(text: str, *, required: bool = False) -> None:
    """Deliver bounded previews without changing a borrowed descriptor's flags."""
    from .diagnostic_writer import deliver
    from .streaming import SEND_TIMEOUT

    delivered = False
    expires = time.monotonic() + SEND_TIMEOUT
    try:
        data = text.encode("utf-8", errors="replace")
        if len(data) > 131_072:
            raise ProtocolError("LIMIT_EXCEEDED", "diagnostic exceeds its bound")
        if isinstance(sys.stderr, io.StringIO):
            sys.stderr.write(text)
            return
        descriptor = sys.stderr.fileno()
        if type(descriptor) is not int:
            raise TypeError("unusable diagnostic descriptor")
        delivered = deliver(descriptor, data, expires)
    except (AttributeError, OSError, TypeError, ValueError, ProtocolError):
        pass
    if required and not delivered:
        raise ProtocolError("SLOW_CONSUMER", "transcript preview receiver is blocked or closed")


class _ArgumentParser(argparse.ArgumentParser):
    def _print_message(self, message, file=None):
        if file is sys.stderr:
            if message:
                _stderr_write(message)
        else:
            super()._print_message(message, file)


def parser() -> argparse.ArgumentParser:
    candidate = _ArgumentParser(prog="kilix-transcribe")
    commands = candidate.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        subparser = commands.add_parser(command)
        if command == "serve":
            subparser.add_argument("--runtime-root", type=Path)
            subparser.add_argument("--installed-asset")
            subparser.add_argument("--content-root", type=Path)
            subparser.add_argument("--model-snapshot-bytes", type=int)
            subparser.add_argument("--lease-device")
            subparser.add_argument("--lease-namespace")
        elif command in {"file", "record"}:
            if command == "file":
                subparser.add_argument("input", type=Path, nargs="?")
            else:
                subparser.add_argument("--seconds", type=float,
                                       help="explicitly record for at most 1 to 120 seconds")
                subparser.add_argument("--device", default="default")
                subparser.add_argument("--vad", action="store_true",
                                       help="end after speech and trailing silence using the legacy energy detector")
            subparser.add_argument("--format", choices=("text", "json", "webvtt", "srt"), default="text")
            subparser.add_argument("--language")
            subparser.add_argument("--task", choices=("transcribe", "translate", "diarize"), default="transcribe")
            subparser.add_argument("--timeout", type=float, default=300)
            subparser.add_argument("--stream", action="store_true",
                                   help="show provisional transcript revisions on stderr during recognition")
        elif command == "cancel":
            subparser.add_argument("job_id", nargs="?")
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        from .runtime import InstalledRuntime, MAX_INPUT_BYTES
        from .service import Service, client_request, request_value, runtime_directory, negotiate_stream
        from .streaming import STREAM_FEATURE
        def show_update(update):
            from .surface import _segment_payload
            _stderr_write("KILIX_TRANSCRIBE_UPDATE " + json.dumps(_segment_payload(update),
                ensure_ascii=False, separators=(",", ":")) + "\n", required=True)
        if arguments.command == "serve":
            from .owned import from_options as execution_options
            execution_policy = execution_options(arguments)

        if arguments.command == "serve":
            from .content import from_options
            model_source = from_options(arguments, provider="kilix-transcribe",
                                        consumer_schema="kilix.transcribe.runtime")
        if arguments.command == "serve" and arguments.runtime_root is not None:
            with model_source if model_source is not None else nullcontext():
                runtime = InstalledRuntime(arguments.runtime_root, model_source=model_source)
                if model_source is not None:
                    # Refuse missing consent or damaged installed bytes before
                    # publishing a socket. Each job still rechecks its model.
                    with model_source.open(lambda: None):
                        pass
                service = Service(runtime, runtime_directory(), execution_policy=execution_policy)
                for sig in (signal.SIGINT, signal.SIGTERM):
                    signal.signal(sig, lambda _sig, _frame: service.stop())
                service.serve()
            return 0
        if arguments.command == "record" and arguments.seconds is not None:
            import threading
            from .recording import record_and_transcribe, receiver_closed
            cancellation = threading.Event()
            previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
            for sig in previous:
                signal.signal(sig, lambda _sig, _frame: cancellation.set())
            indication_lock = threading.Lock()
            def indicate(phase):
                with indication_lock:
                    _stderr_write("KILIX_MICROPHONE " + phase + "\n")
            try:
                payload = record_and_transcribe(runtime_directory(), seconds=arguments.seconds,
                    indicator=indicate, task=arguments.task, language=arguments.language,
                    output=arguments.format, timeout=arguments.timeout, cancelled=cancellation.is_set,
                    disconnected=lambda: receiver_closed(sys.stdout.fileno()), vad=arguments.vad,
                    on_update=show_update if arguments.stream else None,
                    config={"audio": {"device_in": arguments.device}})
                print(payload["output"], end="")
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
            return 0
        if arguments.command == "file" and arguments.input is not None:
            started = time.monotonic()
            descriptor = os.open(arguments.input, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                info = os.fstat(descriptor)
                import stat
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                        or not 0 < info.st_size <= MAX_INPUT_BYTES):
                    raise ProtocolError("INVALID_REQUEST", "input must be a bounded owned regular file")
                header = os.pread(descriptor, 12, 0)
                if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
                    media_type = "audio/wav"
                elif header[:4] == b"fLaC":
                    media_type = "audio/flac"
                elif header[:4] == b"OggS":
                    media_type = "audio/ogg"
                else:
                    raise ProtocolError("UNSUPPORTED_CAPABILITY", "input must be WAV, FLAC, or Ogg audio")
                digest = hashlib.sha256()
                offset = 0
                while offset < info.st_size:
                    chunk = os.pread(descriptor, min(1024 * 1024, info.st_size - offset), offset)
                    if not chunk:
                        raise ProtocolError("INVALID_REQUEST", "input changed while reading")
                    digest.update(chunk)
                    offset += len(chunk)
                args = {"task": arguments.task, "language": arguments.language,
                        "output": arguments.format, "audio_fd": 0,
                        "audio": {"byte_length": info.st_size, "sha256": digest.hexdigest(),
                                  "media_type": media_type}}
                if arguments.stream:
                    negotiate_stream(runtime_directory(), timeout=min(1, arguments.timeout))
                    args["stream"] = STREAM_FEATURE
                remaining = arguments.timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise ProtocolError("DEADLINE_EXCEEDED", "file job deadline expired")
                value = request_value("submit", job_id=uuid.uuid4().hex,
                                      args=args, timeout=remaining)
                payload = client_request(runtime_directory(), value, descriptor,
                                         on_update=show_update if arguments.stream else None)
                print(payload["output"], end="")
                return 0
            finally:
                os.close(descriptor)
        if arguments.command in {"models", "status", "unload", "cancel"}:
            try:
                if arguments.command == "cancel" and arguments.job_id is None:
                    raise RuntimeUnselected()
                payload = client_request(runtime_directory(), request_value(
                    arguments.command, job_id=getattr(arguments, "job_id", None), timeout=5))
            except (FileNotFoundError, ConnectionRefusedError, RuntimeUnselected):
                payload = inspect_command(arguments.command)
            except ProtocolError as error:
                if error.code != "INVALID_RUNTIME":
                    raise
                payload = inspect_command(arguments.command)
        else:
            payload = inspect_command(arguments.command)
    except RuntimeUnselected as error:
        _stderr_write(str(error) + "\n")
        return 69
    except SurfaceError as error:
        _stderr_write(f"KILIX_TRANSCRIBE_REFUSAL [{error.code}] {error}\n")
        return 64
    except (ProtocolError, OSError, ValueError, OverflowError) as error:
        code = error.code if isinstance(error, ProtocolError) else "PROVIDER_UNAVAILABLE"
        _stderr_write(f"KILIX_TRANSCRIBE_REFUSAL [{code}] {error}\n")
        return 69
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    return 0
