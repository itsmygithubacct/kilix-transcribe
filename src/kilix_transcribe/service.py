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
import struct
import copy
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
from .attestation import FEATURE as MANAGED_FEATURE, MINOR as MANAGED_MINOR, ManagedPeer, ManagedResult
from .streaming import STREAM_FEATURE, STREAM_LIMITS, MAX_UPDATES, SEND_TIMEOUT, SegmentStream

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
        self.service_instance = uuid.uuid4().hex
        self.execution_policy = execution_policy
        self._unavailable = False
        self.directory = private_directory(directory)
        self.path = directory / SOCKET_NAME
        self.stopping = threading.Event()
        self.ready = threading.Event()
        self._mutex = threading.Lock()
        self._jobs: dict[str, threading.Event] = {}
        self._job_requests: dict[str, tuple[str, bool, str | None]] = {}
        self._slots = threading.BoundedSemaphore(8)
        self._threads: list[threading.Thread] = []

    def _managed(self, selection=None):
        from .content import InstalledModel
        if (type(self.runtime) is not InstalledRuntime or
                type(self.runtime.model_source) is not InstalledModel):
            raise ProtocolError('INVALID_RUNTIME', 'managed installed model is required')
        if selection is not None and selection['service_instance'] != self.service_instance:
            raise ProtocolError('INCOMPATIBLE_PROTOCOL', 'managed service instance changed')

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
                    self._job_requests.pop(request.job_id, None)
                    claimed = False

        try:
            channel.settimeout(2)
            require_same_uid_peer(channel)
            value, descriptors = receive_packet(channel)
            request = ProviderRequest.from_payload(value)
            deadline = started + request.deadline_ms / 1000
            managed = 'attestation' in request.arguments
            if managed:
                self._managed(request.arguments['attestation'])
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
                    job_instance = uuid.uuid4().hex if managed else None
                    self._job_requests[request.job_id] = (request.request_id, managed, job_instance)
                    claimed = True
            snapshot = verify_request_descriptors(request, descriptors)
            for descriptor in descriptors:
                os.close(descriptor)
            descriptors = ()
            if request.operation == "hello":
                result = {"protocol_major": 1, "protocol_minor": 0}
                if request.arguments.get('features') == [MANAGED_FEATURE]:
                    self._managed()
                    result.update(protocol_minor=MANAGED_MINOR, features=[MANAGED_FEATURE],
                                  service_instance=self.service_instance)
                elif "features" in request.arguments:
                    result.update(protocol_minor=1, features=[STREAM_FEATURE], limits=STREAM_LIMITS)
                kind = "hello"
            elif request.operation == "models":
                record = self.runtime.model_record()
                if managed:
                    record.update(model_sha256=self.runtime.manifest['files']['model.bin'], receipt_enforced=True)
                result = {"models": [record]}
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
                        target_id, target_managed, target_instance = self._job_requests[request.job_id]
                        if managed:
                            if (not target_managed or request.arguments['target_request_id'] != target_id
                                    or request.arguments['target_job_instance'] != target_instance):
                                raise ProtocolError('STALE_JOB', 'managed cancellation targets a different request')
                        elif target_managed:
                            raise ProtocolError('INCOMPATIBLE_PROTOCOL', 'managed job needs pinned cancellation')
                        cancellation.set()
                # This acknowledges a request; terminal CANCELED is emitted on
                # the original job only after its worker has been reaped.
                result = {"cancel_requested": cancellation is not None}
                if managed:
                    result.update(target_request_id=request.arguments['target_request_id'],
                                  target_job_instance=request.arguments['target_job_instance'],
                                  service_instance=self.service_instance)
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
                streaming = request.arguments.get("stream") == STREAM_FEATURE
                remaining = max(0.001, deadline - time.monotonic())
                channel.settimeout(min(SEND_TIMEOUT, remaining) if streaming else remaining)
                acceptance = {'service_instance': self.service_instance, 'job_instance': job_instance} if managed else (
                    {"stream": STREAM_FEATURE, "limits": STREAM_LIMITS} if streaming else {})
                send_packet(channel, _reply(request, "accepted", acceptance))
                def queued(status):
                    remaining = max(0.001, deadline - time.monotonic())
                    channel.settimeout(min(SEND_TIMEOUT, remaining) if streaming else remaining)
                    send_packet(channel, _reply(request, "queued", {
                        "state": status.state, "position": status.position,
                        "lease_version": status.version}))
                def update(value):
                    if cancellation.is_set() or self.stopping.is_set() or _closed(channel):
                        raise ProtocolError("CANCELED", "consumer stopped before update delivery")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ProtocolError("DEADLINE_EXCEEDED", "update delivery deadline exceeded")
                    channel.settimeout(min(SEND_TIMEOUT, remaining))
                    try:
                        send_packet(channel, _reply(request, "update", value))
                    except ProtocolError as error:
                        if cancellation.is_set() or self.stopping.is_set() or _closed(channel):
                            raise ProtocolError("CANCELED", "consumer stopped during update delivery") from error
                        raise ProtocolError("SLOW_CONSUMER", "consumer cannot accept bounded updates") from error
                result = run_job(self.runtime, snapshot, request.arguments,
                                 deadline=deadline, cancel=cancellation,
                                 execution_policy=self.execution_policy, job_id=request.job_id, progress=queued,
                                 updates=update if streaming else None,
                                 disconnected=lambda: self.stopping.is_set() or _closed(channel),
                                 **({'request_id': request.request_id, 'job_instance': job_instance} if managed else {}))
                if managed:
                    if not isinstance(result, ManagedResult):
                        raise ProtocolError('INVALID_RUNTIME', 'managed job evidence is absent')
                    def delivery_check():
                        if cancellation.is_set() or self.stopping.is_set() or _closed(channel):
                            raise ProtocolError('CANCELED', 'consumer stopped before managed delivery')
                        if time.monotonic() >= deadline:
                            raise ProtocolError('DEADLINE_EXCEEDED', 'managed delivery deadline exceeded')
                    result.recheck(delivery_check)
                # Long transcripts use the same bounded descriptor mechanism
                # as audio. No transcript is retained after the connection.
                payload = json.dumps(result, allow_nan=False, ensure_ascii=False,
                                     separators=(",", ":"), sort_keys=True).encode()
                if len(payload) > MAX_RESULT_BYTES:
                    raise ProtocolError('LIMIT_EXCEEDED', 'transcript exceeds its bound')
                if streaming:
                    channel.settimeout(min(SEND_TIMEOUT, max(0.001, deadline - time.monotonic())))
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
                        if managed:
                            metadata.update({k: result[k] for k in ('model_sha256', 'receipt_enforced', 'attestation')})
                            # Use the job snapshot identities, never mutable post-job runtime state.
                            metadata.update(engine_id=result['engine']['id'], model_id=result['model']['id'],
                                            engine_revision=result['engine_revision'], model_revision=result['model_revision'])
                            result.recheck(delivery_check)
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


def negotiate_stream(directory: Path, *, timeout=1):
    """Refuse an older peer before handing it a descriptor for streaming."""
    try:
        response = client_request(directory, request_value("hello", timeout=timeout, args={
            "protocol_major": 1, "protocol_minor": 1, "features": [STREAM_FEATURE]}))
    except ProtocolError as error:
        if error.code in {"INVALID_REQUEST", "INCOMPATIBLE_PROTOCOL"}:
            raise ProtocolError("INCOMPATIBLE_PROTOCOL", "peer does not support transcript streaming") from error
        raise
    if (response != {"protocol_major": 1, "protocol_minor": 1,
                     "features": [STREAM_FEATURE], "limits": STREAM_LIMITS}
            or any(type(response.get(name)) is not int for name in ("protocol_major", "protocol_minor"))):
        raise ProtocolError("INCOMPATIBLE_PROTOCOL", "peer does not support the bounded streaming feature")
    return response


def _endpoint(path):
    try:
        info = path.lstat()
    except OSError as error:
        raise ProtocolError('UNAUTHORIZED_PEER', 'provider endpoint disappeared') from error
    return info.st_dev, info.st_ino, info.st_ctime_ns


def negotiate_managed(directory: Path, *, timeout=1) -> ManagedPeer:
    """Pin negotiation to this endpoint inode, peer credentials and service instance."""
    peer = []
    try:
        response = client_request(directory, request_value('hello', timeout=timeout, args={
            'protocol_major': 1, 'protocol_minor': MANAGED_MINOR, 'features': [MANAGED_FEATURE]}),
            _peer_info=peer)
    except ProtocolError as error:
        if error.code in {'INVALID_REQUEST', 'INCOMPATIBLE_PROTOCOL'}:
            raise ProtocolError('INCOMPATIBLE_PROTOCOL', 'managed-job feature unavailable') from error
        raise
    from .attestation import selection
    if (set(response) != {'protocol_major', 'protocol_minor', 'features', 'service_instance'} or
            type(response.get('protocol_major')) is not int or response['protocol_major'] != 1 or
            type(response.get('protocol_minor')) is not int or response['protocol_minor'] != MANAGED_MINOR or
            response.get('features') != [MANAGED_FEATURE]):
        raise ProtocolError('INCOMPATIBLE_PROTOCOL', 'managed-job negotiation failed')
    selection({'feature': MANAGED_FEATURE, 'service_instance': response['service_instance']})
    return ManagedPeer(str(directory), peer[0][0], peer[0][1], response['service_instance'])


def client_request(directory: Path, value: dict, descriptor: int | None = None, *, cancelled=None, on_update=None,
                   managed: ManagedPeer | None = None, cancel_target: dict | None = None, _peer_info=None) -> dict:
    ProviderRequest.from_payload(value)
    if cancel_target is not None and (managed is None or value['op'] != 'cancel'):
        raise ProtocolError('INVALID_REQUEST', 'cancel target requires a managed cancellation')
    if managed is not None:
        if type(managed) is not ManagedPeer or managed.directory != str(directory) or value.get('op') not in {'submit', 'models', 'cancel'}:
            raise ProtocolError('INVALID_REQUEST', 'managed peer does not bind this request')
        value = copy.deepcopy(value)
        if 'attestation' in value['args']:
            raise ProtocolError('INVALID_REQUEST', 'managed selection is client-owned')
        value['args']['attestation'] = {'feature': MANAGED_FEATURE, 'service_instance': managed.service_instance}
        if value['op'] == 'cancel':
            if type(cancel_target) is not dict or set(cancel_target) != {'request_id', 'job_instance'}:
                raise ProtocolError('INVALID_REQUEST', 'managed cancellation needs its accepted job instance')
            value['args']['target_request_id'] = cancel_target['request_id']
            value['args']['target_job_instance'] = cancel_target['job_instance']
    elif 'attestation' in value.get('args', {}):
        raise ProtocolError('INCOMPATIBLE_PROTOCOL', 'managed requests require explicit peer negotiation')
    ProviderRequest.from_payload(value)
    streaming = value["args"].get("stream") == STREAM_FEATURE
    if on_update is not None and (not callable(on_update) or value["op"] != "submit" or not streaming):
        raise ProtocolError("INVALID_REQUEST", "update callback requires a negotiated streaming submit")
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
    endpoint = _endpoint(path)
    if managed is not None and endpoint != managed.endpoint:
        raise ProtocolError('UNAUTHORIZED_PEER', 'managed endpoint was replaced')
    deadline = time.monotonic() + value["deadline_ms"] / 1000
    state = SegmentStream(value["args"]["task"]) if streaming else None
    accepted = False
    managed_job = None
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
        credentials = struct.unpack('3i', channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        if _endpoint(path) != endpoint or (managed is not None and credentials != managed.credentials):
            raise ProtocolError('UNAUTHORIZED_PEER', 'managed endpoint identity changed')
        if _peer_info is not None:
            _peer_info.append((endpoint, credentials))
        if cancellation_requested():
            raise ProtocolError('CANCELED', 'job canceled before submission')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProtocolError("DEADLINE_EXCEEDED", "provider deadline exceeded before submission")
        channel.settimeout(min(0.2, remaining) if cancelled is not None else remaining)
        send_packet(channel, value, descriptor)
        for _ in range(MAX_UPDATES + 256 if streaming else 256):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProtocolError("DEADLINE_EXCEEDED", "provider deadline exceeded")
            if cancelled is not None:
                while True:
                    if cancellation_requested():
                        remaining = deadline - time.monotonic()
                        if remaining < 0.001:
                            raise ProtocolError("DEADLINE_EXCEEDED", "provider cancellation deadline exceeded")
                        if managed is not None and _endpoint(path) != managed.endpoint:
                            raise ProtocolError('UNAUTHORIZED_PEER', 'managed endpoint changed before cancellation')
                        try:
                            if managed is None or managed_job is not None:
                                target = {'request_id': value['request_id'], 'job_instance': managed_job}
                                client_request(directory, request_value("cancel", job_id=value["job_id"],
                                               timeout=min(0.2, remaining)),
                                               **({'managed': managed, 'cancel_target': target}
                                                  if managed is not None else {}))
                        except (ProtocolError, OSError):
                            if managed is not None:
                                # A changed/unverifiable peer is a refusal, never
                                # a legacy fallback or an automatic renegotiation.
                                raise
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
            try:
                event, descriptors = receive_packet(channel)
            except ProtocolError as error:
                if error.code == 'TRANSPORT_ERROR' and time.monotonic() >= deadline:
                    raise ProtocolError('DEADLINE_EXCEEDED','provider receive deadline exceeded') from error
                raise
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
                if event["type"] == "update":
                    if not streaming or not accepted or descriptors:
                        raise ProtocolError("INVALID_RESPONSE", "unnegotiated transcript update")
                    update = state.apply(event.get("result"))
                    if cancellation_requested():
                        raise ProtocolError("CANCELED", "job canceled before update delivery")
                    if time.monotonic() >= deadline:
                        raise ProtocolError("DEADLINE_EXCEEDED", "update delivery deadline exceeded")
                    if on_update is not None:
                        on_update(update)
                    continue
                if event.get("type") in {"accepted", "queued", "loading", "progress"}:
                    if descriptors or value["op"] != "submit":
                        raise ProtocolError("DESCRIPTOR_MISMATCH", "unexpected progress descriptor")
                    if managed is not None:
                        if event['type'] == 'accepted':
                            acknowledgement = event.get('result')
                            if (managed_job is not None or type(acknowledgement) is not dict
                                    or set(acknowledgement) != {'service_instance', 'job_instance'}
                                    or acknowledgement['service_instance'] != managed.service_instance):
                                raise ProtocolError('INVALID_RESPONSE', 'invalid managed job acceptance')
                            token = acknowledgement['job_instance']
                            if type(token) is not str or len(token) != 32 or any(c not in '0123456789abcdef' for c in token):
                                raise ProtocolError('INVALID_RESPONSE', 'invalid managed job instance')
                            managed_job = token
                        elif managed_job is None:
                            raise ProtocolError('INVALID_RESPONSE', 'managed progress precedes acceptance')
                    if streaming and event["type"] == "accepted":
                        if accepted or event.get("result") != {"stream": STREAM_FEATURE, "limits": STREAM_LIMITS}:
                            raise ProtocolError("INCOMPATIBLE_PROTOCOL", "peer did not acknowledge streaming")
                        accepted = True
                    continue
                result = event.get("result")
                if type(result) is not dict:
                    raise ProtocolError("INVALID_RESPONSE", "invalid result envelope")
                if event.get("type") == "result":
                    if value["op"] != "submit":
                        raise ProtocolError("INVALID_RESPONSE", "unexpected transcript result")
                    population = {"transcript", "engine_id", "engine_revision", "model_id", "model_revision", "duration_ms"}
                    if managed is not None:
                        population.update({'model_sha256', 'receipt_enforced', 'attestation'})
                    if set(result) != population:
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
                    decoded = decode_result(payload, result, value["args"],
                                            **({'request_id': value['request_id'], 'job_id': value.get('job_id'), 'job_instance': managed_job}
                                               if managed is not None else {}))
                    if streaming:
                        if not accepted:
                            raise ProtocolError("INCOMPATIBLE_PROTOCOL", "missing streaming acceptance")
                        state.finish(decoded)
                    if cancellation_requested():
                        raise ProtocolError("CANCELED", "job canceled before delivery")
                    if time.monotonic() >= deadline:
                        raise ProtocolError("DEADLINE_EXCEEDED", "provider deadline exceeded before delivery")
                    if managed is not None and _endpoint(path) != managed.endpoint:
                        raise ProtocolError('UNAUTHORIZED_PEER', 'managed endpoint changed before delivery')
                    return decoded
                expected = {"hello": "hello", "status": "status", "models": "models",
                            "unload": "unloaded", "cancel": "canceled"}.get(value["op"])
                if descriptors or event.get("type") != expected:
                    raise ProtocolError("INVALID_RESPONSE", "unexpected provider response")
                if time.monotonic() >= deadline:
                    raise ProtocolError("DEADLINE_EXCEEDED", "provider deadline exceeded before delivery")
                if managed is not None and value['op'] == 'cancel':
                    if (set(result) != {'cancel_requested', 'target_request_id', 'target_job_instance', 'service_instance'}
                            or type(result['cancel_requested']) is not bool
                            # Bind to the private, validated request that was sent,
                            # never the caller's potentially mutated mapping.
                            or result['target_request_id'] != value['args']['target_request_id']
                            or result['target_job_instance'] != value['args']['target_job_instance']
                            or result['service_instance'] != managed.service_instance):
                        raise ProtocolError('INVALID_RESPONSE', 'unbound managed cancellation acknowledgement')
                if (managed is not None or _peer_info is not None) and _endpoint(path) != endpoint:
                    raise ProtocolError('UNAUTHORIZED_PEER', 'endpoint changed before delivery')
                return result
            finally:
                for received in descriptors:
                    os.close(received)
    raise ProtocolError("LIMIT_EXCEEDED", "provider event limit exceeded")
