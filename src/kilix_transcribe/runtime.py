"""An installed, digest-bound whisper.cpp runtime and supervised jobs.

The installation manifest names files below the provider's own installation.
It is read only at service startup; job requests cannot choose files or tools.
It describes a local runtime, not a qualified release profile or an installer
licence receipt.
"""

from __future__ import annotations

import hashlib
import ctypes
from contextlib import contextmanager
import fcntl
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


def memory_file() -> int:
    # Some managed CPython builds omit os.memfd_create even on a supporting
    # Linux/glibc host. Use the same libc operation, never a writable fallback.
    libc = ctypes.CDLL(None, use_errno=True)
    create = libc.memfd_create
    create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    create.restype = ctypes.c_int
    descriptor = create(b"kilix-runtime", 0x0001 | 0x0002)
    if descriptor < 0:
        raise ProtocolError("INVALID_RUNTIME", "sealed runtime snapshots are unavailable")
    return descriptor


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    with os.fdopen(descriptor, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ProtocolError("INVALID_RUNTIME", "runtime source is not a regular file")
        remaining = info.st_size
        while remaining:
            block = source.read(min(remaining, 1024 * 1024))
            if not block:
                raise ProtocolError("INVALID_RUNTIME", "runtime source ended early")
            digest.update(block)
            remaining -= len(block)
        if os.fstat(source.fileno()).st_size != info.st_size:
            raise ProtocolError("INVALID_RUNTIME", "runtime source size changed")
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
        root_fd, self._directories = self._open_root()
        os.close(root_fd)
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

    def _open_root(self):
        descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        identities = []
        try:
            for component in self.root.parts[1:]:
                child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
                info = os.fstat(descriptor)
                identities.append((info.st_dev, info.st_ino, info.st_mode, info.st_uid))
            return descriptor, identities
        except BaseException:
            os.close(descriptor)
            raise

    @contextmanager
    def snapshots(self, check: Callable[[], None] = lambda: None):
        """Carry verified, sealed bytes through execution; never reopen paths."""
        root_fd, directories = self._open_root()
        descriptors = {}
        try:
            if directories != self._directories:
                raise ProtocolError("INVALID_RUNTIME", "runtime directory identity changed")
            for name, expected in self.manifest["files"].items():
                check()
                source = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=root_fd)
                snapshot = -1
                try:
                    before = os.fstat(source)
                    if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                            or before.st_mode & 0o022 or before.st_size != self._identities[name][2]):
                        raise ProtocolError("INVALID_RUNTIME", "unsafe runtime source")
                    snapshot = memory_file()
                    digest = hashlib.sha256()
                    remaining = before.st_size
                    while remaining:
                        check()
                        block = os.read(source, min(remaining, 1024 * 1024))
                        if not block:
                            raise ProtocolError("INVALID_RUNTIME", "runtime source ended early")
                        digest.update(block)
                        remaining -= len(block)
                        view = memoryview(block)
                        while view:
                            written = os.write(snapshot, view)
                            if written <= 0:
                                raise ProtocolError("INVALID_RUNTIME", "runtime snapshot stalled")
                            view = view[written:]
                    if digest.hexdigest() != expected or os.fstat(source).st_size != before.st_size:
                        raise ProtocolError("INVALID_RUNTIME", "runtime source digest mismatch")
                    os.fchmod(snapshot, 0o400 if name == "model.bin" else 0o500)
                    # Linux UAPI values; managed Python can also omit these
                    # fcntl names. F_ADD_SEALS=1024+9, WRITE|GROW|SHRINK|SEAL.
                    fcntl.fcntl(snapshot, 1033, 0x0008 | 0x0004 | 0x0002 | 0x0001)
                    reader = os.open(f"/proc/self/fd/{snapshot}", os.O_RDONLY | os.O_CLOEXEC)
                    descriptors[name] = reader
                finally:
                    os.close(source)
                    if snapshot >= 0:
                        os.close(snapshot)
            yield descriptors
        finally:
            os.close(root_fd)
            for descriptor in descriptors.values():
                os.close(descriptor)

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
    """Wait for the dedicated supervisor to reap every owned descendant."""
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.wait()
        raise ProtocolError("SUPERVISOR_FAILED", "descendant cleanup did not complete") from error


def run_job(runtime: InstalledRuntime, audio_fd: int, args: dict, *,
            deadline: float, cancel: threading.Event,
            disconnected: Callable[[], bool] = lambda: False) -> dict:
    if cancel.is_set() or disconnected():
        raise ProtocolError("CANCELED", "job canceled")
    if time.monotonic() >= deadline:
        raise ProtocolError("DEADLINE_EXCEEDED", "job deadline exceeded")
    runtime.verify_unchanged()
    def check():
        if cancel.is_set() or disconnected():
            raise ProtocolError("CANCELED", "job canceled")
        if time.monotonic() >= deadline:
            raise ProtocolError("DEADLINE_EXCEEDED", "job deadline exceeded")
    if args["task"] == "diarize":
        raise ProtocolError("UNSUPPORTED_CAPABILITY", "no diarization profile is installed")
    if not 0 < os.fstat(audio_fd).st_size <= MAX_INPUT_BYTES:
        raise ProtocolError("LIMIT_EXCEEDED", "audio exceeds the runtime input bound")
    # File descriptors carry audio; the isolated worker's argv carries no text,
    # transcript, source filename, or model paths supplied by a client.
    worker = Path(__file__).with_name("supervisor.py")
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
                   "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    with runtime.snapshots(check) as runtime_fds, tempfile.TemporaryDirectory(prefix="kilix-transcribe-job-") as workspace, tempfile.TemporaryFile() as output:
        job = {"runtime_fds": runtime_fds, "manifest": runtime.manifest,
               "audio_fd": audio_fd, "args": args,
               "workspace": workspace}
        process = subprocess.Popen(
            [sys.executable, "-I", str(worker)], stdin=subprocess.PIPE,
            stdout=output, stderr=subprocess.DEVNULL, env=environment,
            pass_fds=(audio_fd, *runtime_fds.values()), start_new_session=True,
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
