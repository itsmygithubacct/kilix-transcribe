"""Local transcription client and installed-runtime service entry point."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import uuid
from collections.abc import Sequence

from .surface import COMMANDS, RuntimeUnselected, SurfaceError, inspect_command
from .protocol import ProtocolError


def parser() -> argparse.ArgumentParser:
    candidate = argparse.ArgumentParser(prog="kilix-transcribe")
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
        elif command == "file":
            subparser.add_argument("input", type=Path, nargs="?")
            subparser.add_argument("--format", choices=("text", "json", "webvtt", "srt"), default="text")
            subparser.add_argument("--language")
            subparser.add_argument("--task", choices=("transcribe", "translate", "diarize"), default="transcribe")
            subparser.add_argument("--timeout", type=float, default=300)
        elif command == "cancel":
            subparser.add_argument("job_id", nargs="?")
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        from .runtime import InstalledRuntime, MAX_INPUT_BYTES
        from .service import Service, client_request, request_value, runtime_directory
        if arguments.command == "serve":
            from .owned import from_options as execution_options
            execution_policy = execution_options(arguments)

        if arguments.command == "serve":
            from .content import from_options
            model_source = from_options(arguments, provider="kilix-transcribe",
                                        consumer_schema="kilix.transcribe.runtime")
        if arguments.command == "serve" and arguments.runtime_root is not None:
            with model_source if model_source is not None else nullcontext():
                service = Service(InstalledRuntime(arguments.runtime_root, model_source=model_source), runtime_directory(),
                                  execution_policy=execution_policy)
                for sig in (signal.SIGINT, signal.SIGTERM):
                    signal.signal(sig, lambda _sig, _frame: service.stop())
                service.serve()
            return 0
        if arguments.command == "file" and arguments.input is not None:
            descriptor = os.open(arguments.input, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
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
                value = request_value("submit", job_id=uuid.uuid4().hex,
                                      args=args, timeout=arguments.timeout)
                payload = client_request(runtime_directory(), value, descriptor)
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
        print(str(error), file=sys.stderr)
        return 69
    except SurfaceError as error:
        print(f"KILIX_TRANSCRIBE_REFUSAL [{error.code}] {error}", file=sys.stderr)
        return 64
    except (ProtocolError, OSError, ValueError, OverflowError) as error:
        code = error.code if isinstance(error, ProtocolError) else "PROVIDER_UNAVAILABLE"
        print(f"KILIX_TRANSCRIBE_REFUSAL [{code}] {error}", file=sys.stderr)
        return 69
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    return 0
