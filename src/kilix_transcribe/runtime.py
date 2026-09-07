"""An installed, digest-bound whisper.cpp runtime and supervised jobs.

The installation manifest names files below the provider's own installation.
It is read only at service startup; job requests cannot choose files or tools.
It describes a local runtime, not a qualified release profile or an installer
licence receipt.
"""

from __future__ import annotations

import hashlib
import ctypes
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable

from .protocol import ProtocolError, decode_payload

RUNTIME_SCHEMA = "kilix.transcribe.runtime/v1"
ENGINE_COMMIT = "371b5a7561823ab2bb32142d2751e35e7534727b"
MAX_INPUT_BYTES = 512 * 1024 * 1024
MAX_RESULT_BYTES = 16 * 1024 * 1024
MAX_AUDIO_SECONDS = 3600


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def private_directory(path: Path, *, create: bool = False) -> Path:
    if not path.is_absolute():
        raise ProtocolError("INVALID_RUNTIME", "directory must be absolute")
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or info.st_mode & 0o077):
        raise ProtocolError("INVALID_RUNTIME", "directory must be private and owned")
    return path


class InstalledRuntime:
    def __init__(self, root: Path):
        self.root = private_directory(root)
        manifest = root / "runtime.json"
        info = manifest.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or info.st_mode & 0o022 or info.st_size > 65_536):
            raise ProtocolError("INVALID_RUNTIME", "runtime manifest is unsafe")
        value = decode_payload(manifest.read_bytes())
        if (set(value) != {"schema", "engine_revision", "model", "files"}
                or value["schema"] != RUNTIME_SCHEMA
                or value["engine_revision"] != ENGINE_COMMIT):
            raise ProtocolError("INVALID_RUNTIME", "unsupported runtime identity")
        model = value["model"]
        if (type(model) is not dict or set(model) != {"id", "revision"}
                or any(type(v) is not str or not v or len(v) > 128
                       for v in model.values())):
            raise ProtocolError("INVALID_RUNTIME", "invalid model identity")
        files = value["files"]
        if type(files) is not dict or set(files) != {"whisper-cli", "ffmpeg", "model.bin"}:
            raise ProtocolError("INVALID_RUNTIME", "incomplete runtime files")
        self._identities = {}
        for name, expected in files.items():
            path = root / name
            before = path.lstat()
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                    or before.st_mode & 0o022 or type(expected) is not str
                    or len(expected) != 64 or digest_file(path) != expected):
                raise ProtocolError("INVALID_RUNTIME", "runtime file identity mismatch")
            if name != "model.bin" and not os.access(path, os.X_OK):
                raise ProtocolError("INVALID_RUNTIME", "runtime tool is not executable")
            after = path.lstat()
            identity = self._identity(before)
            if self._identity(after) != identity:
                raise ProtocolError("INVALID_RUNTIME", "runtime changed during validation")
            self._identities[name] = identity
        self.model_id = model["id"]
        self.model_revision = model["revision"]
        self.manifest = value

    @staticmethod
    def _identity(info: os.stat_result) -> tuple[int, ...]:
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
                info.st_ctime_ns, info.st_mode)

    def verify_unchanged(self) -> None:
        for name, identity in self._identities.items():
            path = self.root / name
            # Same-size writes can share a filesystem timestamp. Metadata
            # equality alone is not proof that installed bytes are unchanged.
            if (self._identity(path.lstat()) != identity
                    or digest_file(path) != self.manifest["files"][name]):
                raise ProtocolError("INVALID_RUNTIME", "runtime files changed; restart required")

    def model_record(self) -> dict:
        return {"id": self.model_id, "revision": self.model_revision,
                "engine_id": "whisper.cpp", "engine_revision": ENGINE_COMMIT,
                "installed": True, "release_qualified": False,
                "capabilities": ["transcribe", "translate", "language_id", "word_timestamps"]}


def stop_process(process: subprocess.Popen) -> None:
    """Reap the whole worker group before acknowledging cancellation."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        pass
    # A worker can exit before one of its descendants. Always kill the group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()
    # Linux reparents orphaned grandchildren to this service, so a canceled
    # worker cannot leave engine zombies with an unrelated init process.
    deadline = time.monotonic() + 2
    while True:
        try:
            reaped, _ = os.waitpid(-process.pid, os.WNOHANG)
        except ChildProcessError:
            break
        if reaped == 0:
            if time.monotonic() >= deadline:
                raise ProtocolError("SUPERVISOR_FAILED", "worker group did not terminate")
            time.sleep(0.005)


def run_job(runtime: InstalledRuntime, audio_fd: int, args: dict, *,
            deadline: float, cancel: threading.Event,
            disconnected: Callable[[], bool] = lambda: False) -> dict:
    if cancel.is_set() or disconnected():
        raise ProtocolError("CANCELED", "job canceled")
    if time.monotonic() >= deadline:
        raise ProtocolError("DEADLINE_EXCEEDED", "job deadline exceeded")
    runtime.verify_unchanged()
    # PR_SET_CHILD_SUBREAPER is Linux's process-tree ownership mechanism.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise ProtocolError("SUPERVISOR_FAILED", "cannot own worker descendants")
    if args["task"] == "diarize":
        raise ProtocolError("UNSUPPORTED_CAPABILITY", "no diarization profile is installed")
    if not 0 < os.fstat(audio_fd).st_size <= MAX_INPUT_BYTES:
        raise ProtocolError("LIMIT_EXCEEDED", "audio exceeds the runtime input bound")
    # File descriptors carry audio; the isolated worker's argv carries no text,
    # transcript, source filename, or model paths supplied by a client.
    worker = Path(__file__).with_name("worker.py")
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
                   "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    with tempfile.TemporaryDirectory(prefix="kilix-transcribe-job-") as workspace, tempfile.TemporaryFile() as output:
        job = {"runtime": str(runtime.root), "manifest": runtime.manifest,
               "audio_fd": audio_fd, "args": args,
               "workspace": workspace}
        process = subprocess.Popen(
            [sys.executable, "-I", str(worker)], stdin=subprocess.PIPE,
            stdout=output, stderr=subprocess.DEVNULL, env=environment,
            pass_fds=(audio_fd,), start_new_session=True,
        )
        try:
            assert process.stdin is not None
            process.stdin.write(json.dumps(job, separators=(",", ":")).encode())
            process.stdin.close()
            while process.poll() is None:
                if cancel.is_set() or disconnected():
                    raise ProtocolError("CANCELED", "job canceled")
                if time.monotonic() >= deadline:
                    raise ProtocolError("DEADLINE_EXCEEDED", "job deadline exceeded")
                cancel.wait(0.025)
            if cancel.is_set() or disconnected():
                raise ProtocolError("CANCELED", "job canceled")
            if time.monotonic() >= deadline:
                raise ProtocolError("DEADLINE_EXCEEDED", "job deadline exceeded")
            if process.returncode != 0:
                raise ProtocolError("ENGINE_FAILED", "speech worker failed")
            output.seek(0)
            payload = output.read(MAX_RESULT_BYTES + 1)
            if len(payload) > MAX_RESULT_BYTES:
                raise ProtocolError("LIMIT_EXCEEDED", "transcript exceeds its bound")
            try:
                result = json.loads(payload)
            except (ValueError, UnicodeDecodeError) as error:
                raise ProtocolError("ENGINE_FAILED", "invalid worker result") from error
            if type(result) is not dict or result.get("engine_revision") != ENGINE_COMMIT:
                raise ProtocolError("ENGINE_FAILED", "unbound worker result")
            return result
        finally:
            stop_process(process)
