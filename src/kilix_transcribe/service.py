"""Private seqpacket service for supervised, one-at-a-time transcription."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import socket
import stat
import tempfile
import threading
import time
import uuid

from .protocol import (
    PROTOCOL_SCHEMA, ProtocolError, ProviderRequest, receive_packet,
    require_same_uid_peer, send_packet, verify_request_descriptors,
)
from .runtime import InstalledRuntime, MAX_INPUT_BYTES, MAX_RESULT_BYTES, private_directory, run_job
from .results import decode_result

SOCKET_NAME = "kilix-transcribe.sock"


def runtime_directory() -> Path:
    value = os.environ.get("XDG_RUNTIME_DIR")
    if not value:
        raise ProtocolError("INVALID_RUNTIME", "XDG_RUNTIME_DIR is required")
    return private_directory(Path(value))


def _closed(channel: socket.socket) -> bool:
    try:
        if not select.select([channel], [], [], 0)[0]:
            return False
        # Requests are single-packet operations: extra data is not another job.
        channel.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
        return True
    except BlockingIOError:
        return False
    except OSError:
        return True


def _reply(request: ProviderRequest, kind: str, result: dict) -> dict:
    return {"schema": PROTOCOL_SCHEMA, "type": kind,
            "request_id": request.request_id, "job_id": request.job_id,
            "result": result}


class Service:
    def __init__(self, runtime: InstalledRuntime, directory: Path, *, execution_policy=None):
        self.runtime = runtime
        self.execution_policy = execution_policy
        self._unavailable = False
        self.directory = private_directory(directory)
        self.path = directory / SOCKET_NAME
        self.stopping = threading.Event()
        self.ready = threading.Event()
        self._mutex = threading.Lock()
        self._jobs: dict[str, threading.Event] = {}
        self._slots = threading.BoundedSemaphore(8)
        self._threads: list[threading.Thread] = []

    def stop(self) -> None:
        self.stopping.set()
        with self._mutex:
            for cancellation in self._jobs.values():
                cancellation.set()

    def serve(self) -> None:
        if len(os.fsencode(self.path)) >= 108:
            raise ProtocolError("INVALID_RUNTIME", "socket path is too long")
        lock_fd = os.open(str(self.path) + ".lock", os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        bound = None
        try:
            info = os.fstat(lock_fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise ProtocolError("INVALID_RUNTIME", "unsafe service lock")
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ProtocolError("BUSY", "provider is already running") from error
            try:
                old = self.path.lstat()
            except FileNotFoundError:
                old = None
            if old is not None:
                if not stat.S_ISSOCK(old.st_mode) or old.st_uid != os.geteuid():
                    raise ProtocolError("INVALID_RUNTIME", "refusing existing non-socket path")
                # Do not replace a live endpoint created by another service.
                probe = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
                try:
                    probe.settimeout(0.1)
                    probe.connect(str(self.path))
                except ConnectionRefusedError:
                    self.path.unlink()
                else:
                    raise ProtocolError("BUSY", "provider endpoint is live")
                finally:
                    probe.close()
            listener.bind(str(self.path))
            os.chmod(self.path, 0o600)
            current = self.path.lstat()
            bound = current.st_dev, current.st_ino
            listener.listen(8)
            listener.settimeout(0.2)
            self.ready.set()
            while not self.stopping.is_set():
                try:
                    channel, _ = listener.accept()
                except TimeoutError:
                    continue
                if not self._slots.acquire(blocking=False):
                    channel.close()
                    continue
                self._threads = [thread for thread in self._threads if thread.is_alive()]
                thread = threading.Thread(target=self._handle, args=(channel,), daemon=True)
                self._threads.append(thread)
                thread.start()
        finally:
            self.stop()
            listener.close()
            for thread in self._threads:
                thread.join()
            if bound is not None:
                try:
                    current = self.path.lstat()
                    if (current.st_dev, current.st_ino) == bound:
                        self.path.unlink()
                except FileNotFoundError:
                    pass
            os.close(lock_fd)

    def _handle(self, channel: socket.socket) -> None:
        request = None
        descriptors = ()
        snapshot = None
        claimed = False
        started = time.monotonic()

        def finish_job():
            # A terminal event promises that another job can claim the worker.
            # Releasing here also prevents a suspended old sender's finally
            # from removing a later job that reuses the same client job ID.
            nonlocal snapshot, descriptors, claimed
            if snapshot is not None:
                os.close(snapshot)
                snapshot = None
            for descriptor in descriptors:
                os.close(descriptor)
            descriptors = ()
            if claimed:
                with self._mutex:
                    self._jobs.pop(request.job_id, None)
                    claimed = False

        try:
            channel.settimeout(2)
            require_same_uid_peer(channel)
            value, descriptors = receive_packet(channel)
            request = ProviderRequest.from_payload(value)
            deadline = started + request.deadline_ms / 1000
            if request.operation == "submit":
                if request.arguments["audio"]["byte_length"] > MAX_INPUT_BYTES:
                    raise ProtocolError("LIMIT_EXCEEDED", "input exceeds runtime bound")
                with self._mutex:
                    if self._unavailable:
                        raise ProtocolError("SUPERVISOR_FAILED", "owned cleanup remains unproven")
                    if self._jobs or self.stopping.is_set():
                        raise ProtocolError("BUSY", "speech worker is busy")
                    cancellation = threading.Event()
                    self._jobs[request.job_id] = cancellation
                    claimed = True
            snapshot = verify_request_descriptors(request, descriptors)
            for descriptor in descriptors:
                os.close(descriptor)
            descriptors = ()
            if request.operation == "hello":
                result = {"protocol_major": 1, "protocol_minor": 0}
                kind = "hello"
            elif request.operation == "models":
                result = {"models": [self.runtime.model_record()]}
                kind = "models"
            elif request.operation == "status":
                with self._mutex:
                    busy = bool(self._jobs)
                    unavailable = self._unavailable
                result = {"provider_state": "unavailable" if unavailable else "busy" if busy else "ready", "worker_active": busy,
                          "engine_id": "whisper.cpp", "model_id": self.runtime.model_id,
                          "release_qualified": False}
                kind = "status"
            elif request.operation == "cancel":
                with self._mutex:
                    cancellation = self._jobs.get(request.job_id)
                    if cancellation is not None:
                        cancellation.set()
                # This acknowledges a request; terminal CANCELED is emitted on
                # the original job only after its worker has been reaped.
                result = {"cancel_requested": cancellation is not None}
                kind = "canceled"
            elif request.operation == "unload":
                with self._mutex:
                    if self._unavailable:
                        raise ProtocolError("SUPERVISOR_FAILED", "owned cleanup remains unproven")
                    if self._jobs:
                        raise ProtocolError("BUSY", "cannot unload during a job")
                result = {"loaded": False}
                kind = "unloaded"
            else:
                channel.settimeout(max(0.001, deadline - time.monotonic()))
                send_packet(channel, _reply(request, "accepted", {}))
                def queued(status):
                    channel.settimeout(max(0.001, deadline - time.monotonic()))
                    send_packet(channel, _reply(request, "queued", {
                        "state": status.state, "position": status.position,
                        "lease_version": status.version}))
                result = run_job(self.runtime, snapshot, request.arguments,
                                 deadline=deadline, cancel=cancellation,
                                 execution_policy=self.execution_policy, job_id=request.job_id, progress=queued,
                                 disconnected=lambda: self.stopping.is_set() or _closed(channel))
                # Long transcripts use the same bounded descriptor mechanism
                # as audio. No transcript is retained after the connection.
                payload = json.dumps(result, allow_nan=False, ensure_ascii=False,
                                     separators=(",", ":"), sort_keys=True).encode()
                with tempfile.TemporaryFile() as writer:
                    writer.write(payload)
                    writer.flush()
                    result_fd = os.open(f"/proc/self/fd/{writer.fileno()}", os.O_RDONLY | os.O_CLOEXEC)
                    try:
                        metadata = {"transcript": {"fd": 0, "byte_length": len(payload),
                                                   "sha256": hashlib.sha256(payload).hexdigest()},
                                    "engine_id": "whisper.cpp", "model_id": self.runtime.model_id,
                                    "engine_revision": self.runtime.manifest["engine_revision"],
                                    "model_revision": self.runtime.model_revision,
                                    "duration_ms": result["duration_ms"]}
                        finish_job()
                        send_packet(channel, _reply(request, "result", metadata), result_fd)
                    finally:
                        os.close(result_fd)
                return
            send_packet(channel, _reply(request, kind, result))
        except (ProtocolError, OSError, ValueError, KeyError) as error:
            if isinstance(error, ProtocolError) and error.code == "SUPERVISOR_FAILED":
                with self._mutex:
                    self._unavailable = True
            finish_job()
            if request is not None:
                code = error.code if isinstance(error, ProtocolError) else "PROVIDER_ERROR"
                event = _reply(request, "error", {})
                event.pop("result")
                event["error"] = {"code": code}
                try:
                    channel.settimeout(0.2)
                    send_packet(channel, event)
                except (ProtocolError, OSError):
                    pass
        finally:
            finish_job()
            channel.close()
            self._slots.release()


def request_value(operation: str, *, job_id: str | None = None,
                  args: dict | None = None, timeout: float = 300) -> dict:
    value = {"schema": PROTOCOL_SCHEMA, "type": "request", "request_id": uuid.uuid4().hex,
             "op": operation, "deadline_ms": int(timeout * 1000), "args": args or {}}
    if job_id is not None:
        value["job_id"] = job_id
    ProviderRequest.from_payload(value)
    return value


def client_request(directory: Path, value: dict, descriptor: int | None = None, *, cancelled=None) -> dict:
    ProviderRequest.from_payload(value)
    if cancelled is not None and (not callable(cancelled) or value['op'] != 'submit'):
        raise ProtocolError('INVALID_REQUEST', 'cancellation callback requires a submit request')

    def cancellation_requested():
        if cancelled is None:
            return False
        decision = cancelled()
        if type(decision) is not bool:
            raise ProtocolError('INVALID_REQUEST', 'cancellation callback must return boolean')
        return decision

    if cancellation_requested():
        raise ProtocolError('CANCELED', 'job canceled before submission')
    private_directory(directory)
    path = directory / SOCKET_NAME
    info = path.lstat()
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o177:
        raise ProtocolError("UNAUTHORIZED_PEER", "unsafe provider endpoint")
    deadline = time.monotonic() + value["deadline_ms"] / 1000
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as channel:
        remaining = max(0.001, deadline - time.monotonic())
        channel.settimeout(min(0.2, remaining) if cancelled is not None else remaining)
        try:
            channel.connect(str(path))
        except OSError as error:
            if cancellation_requested():
                raise ProtocolError('CANCELED', 'job canceled before submission') from error
            raise ProtocolError('TRANSPORT_ERROR', 'provider connection failed') from error
        require_same_uid_peer(channel)
        if cancellation_requested():
            raise ProtocolError('CANCELED', 'job canceled before submission')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProtocolError("DEADLINE_EXCEEDED", "provider deadline exceeded before submission")
        channel.settimeout(min(0.2, remaining) if cancelled is not None else remaining)
        send_packet(channel, value, descriptor)
        for _ in range(256):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProtocolError("DEADLINE_EXCEEDED", "provider deadline exceeded")
            if cancelled is not None:
                while True:
                    if cancellation_requested():
                        remaining = deadline - time.monotonic()
                        if remaining < 0.001:
                            raise ProtocolError("DEADLINE_EXCEEDED", "provider cancellation deadline exceeded")
                        try:
                            client_request(directory, request_value("cancel", job_id=value["job_id"],
                                           timeout=min(0.2, remaining)))
                        except (ProtocolError, OSError):
                            pass
                        # Return closes our original channel and descriptors.
                        # Neither local cancellation nor an ACK proves that
                        # the provider has finished its authenticated teardown.
                        raise ProtocolError("CANCELED", "client canceled; provider cleanup pending")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ProtocolError("DEADLINE_EXCEEDED", "provider cancellation deadline exceeded")
                    if select.select([channel], [], [], min(0.05, remaining))[0]:
                        break
            channel.settimeout(remaining)
            event, descriptors = receive_packet(channel)
            try:
                if (event.get("schema") != PROTOCOL_SCHEMA
                        or event.get("request_id") != value["request_id"]
                        or event.get("job_id") != value.get("job_id")):
                    raise ProtocolError("INVALID_RESPONSE", "unbound provider response")
                if type(event.get("type")) is not str:
                    raise ProtocolError("INVALID_RESPONSE", "invalid event type")
                if event.get("type") == "error":
                    error = event.get("error")
                    if type(error) is not dict or descriptors:
                        raise ProtocolError("INVALID_RESPONSE", "invalid error response")
                    code = error.get("code", "PROVIDER_ERROR")
                    if (type(code) is not str or not 0 < len(code) <= 64
                            or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for c in code)):
                        code = "PROVIDER_ERROR"
                    raise ProtocolError(code, "provider refused the request")
                if event.get("type") in {"accepted", "queued", "loading", "progress"}:
                    if descriptors or value["op"] != "submit":
                        raise ProtocolError("DESCRIPTOR_MISMATCH", "unexpected progress descriptor")
                    continue
                result = event.get("result")
                if type(result) is not dict:
                    raise ProtocolError("INVALID_RESPONSE", "invalid result envelope")
                if event.get("type") == "result":
                    if value["op"] != "submit":
                        raise ProtocolError("INVALID_RESPONSE", "unexpected transcript result")
                    if set(result) != {"transcript", "engine_id", "engine_revision", "model_id", "model_revision", "duration_ms"}:
                        raise ProtocolError("INVALID_RESPONSE", "invalid result metadata population")
                    metadata = result.get("transcript")
                    if (len(descriptors) != 1 or type(metadata) is not dict
                            or set(metadata) != {"fd", "byte_length", "sha256"}
                            or type(metadata.get("fd")) is not int or metadata["fd"] != 0
                            or type(metadata.get("byte_length")) is not int
                            or not 0 < metadata["byte_length"] <= MAX_RESULT_BYTES
                            or type(metadata.get("sha256")) is not str
                            or len(metadata["sha256"]) != 64
                            or any(c not in "0123456789abcdef" for c in metadata["sha256"])):
                        raise ProtocolError("DESCRIPTOR_MISMATCH", "invalid transcript descriptor")
                    fd = descriptors[0]
                    info = os.fstat(fd)
                    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                            or info.st_size != metadata["byte_length"]
                            or fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY):
                        raise ProtocolError("DESCRIPTOR_MISMATCH", "invalid transcript file")
                    payload = os.pread(fd, MAX_RESULT_BYTES + 1, 0)
                    if len(payload) != info.st_size or hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
                        raise ProtocolError("DESCRIPTOR_MISMATCH", "transcript digest mismatch")
                    decoded = decode_result(payload, result, value["args"])
                    if cancellation_requested():
                        raise ProtocolError("CANCELED", "job canceled before delivery")
                    return decoded
                expected = {"hello": "hello", "status": "status", "models": "models",
                            "unload": "unloaded", "cancel": "canceled"}.get(value["op"])
                if descriptors or event.get("type") != expected:
                    raise ProtocolError("INVALID_RESPONSE", "unexpected provider response")
                return result
            finally:
                for received in descriptors:
                    os.close(received)
    raise ProtocolError("LIMIT_EXCEEDED", "provider event limit exceeded")
