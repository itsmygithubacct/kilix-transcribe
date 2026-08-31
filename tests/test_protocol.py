from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path

from kilix_transcribe.output import AtomicTranscriptStore
from kilix_transcribe.protocol import (
    PROTOCOL_SCHEMA,
    ProtocolError,
    ProviderRequest,
    decode_payload,
    encode_payload,
    frame_payload,
    receive_packet,
    require_same_uid_peer,
    send_packet,
    unframe_payload,
    verify_request_descriptors,
)
from kilix_transcribe.surface import SegmentUpdate, TranscriptAssembler


def submit_payload() -> dict:
    return {
        "schema": PROTOCOL_SCHEMA,
        "type": "request",
        "request_id": "request-1",
        "op": "submit",
        "job_id": "job-1",
        "deadline_ms": 30_000,
        "args": {
            "task": "transcribe",
            "language": "en",
            "output": "json",
            "audio_fd": 0,
            "audio": {
                "byte_length": 4,
                "sha256": hashlib.sha256(b"RIFF").hexdigest(),
                "media_type": "audio/wav",
            },
        },
    }


class DecodeTests(unittest.TestCase):
    def refusal(self, payload: bytes, code: str) -> None:
        with self.assertRaises(ProtocolError) as caught:
            decode_payload(payload)
        self.assertEqual(caught.exception.code, code)

    def test_round_trip_is_canonical(self) -> None:
        value = submit_payload()
        encoded = encode_payload(value)
        self.assertEqual(encode_payload(decode_payload(encoded)), encoded)
        self.assertEqual(unframe_payload(frame_payload(value)), value)

    def test_duplicate_keys_and_non_finite_numbers_refuse(self) -> None:
        self.refusal(b'{"a":1,"a":2}', "INVALID_REQUEST")
        self.refusal(b'{"a":NaN}', "INVALID_REQUEST")

    def test_extreme_depth_and_integer_refuse_stably(self) -> None:
        nested = b"[" * 10_000 + b"true" + b"]" * 10_000
        self.assertLess(len(nested), 65_536)
        self.refusal(nested, "LIMIT_EXCEEDED")
        integer = b'{"n":' + b"9" * 50_000 + b"}"
        self.assertLess(len(integer), 65_536)
        self.refusal(integer, "LIMIT_EXCEEDED")

    def test_lone_surrogate_refuses_stably(self) -> None:
        self.refusal(b'{"text":"\\ud800"}', "INVALID_REQUEST")


class RequestTests(unittest.TestCase):
    def refusal(self, value: dict, code: str) -> None:
        with self.assertRaises(ProtocolError) as caught:
            ProviderRequest.from_payload(value)
        self.assertEqual(caught.exception.code, code)

    def test_submit_is_typed(self) -> None:
        request = ProviderRequest.from_payload(submit_payload())
        self.assertEqual((request.operation, request.job_id), ("submit", "job-1"))

    def test_forbidden_fields_are_recursive(self) -> None:
        value = submit_payload()
        value["args"]["audio"]["path"] = "/caller/chosen"
        self.refusal(value, "FORBIDDEN_FIELD")

    def test_boolean_numeric_fields_refuse(self) -> None:
        value = submit_payload()
        value["deadline_ms"] = True
        self.refusal(value, "LIMIT_EXCEEDED")
        hello = copy.deepcopy(value)
        hello.pop("job_id")
        hello.update(op="hello", args={"protocol_major": True, "protocol_minor": 0})
        hello["deadline_ms"] = 1
        self.refusal(hello, "INCOMPATIBLE_PROTOCOL")

    def test_non_job_operations_forbid_job_id_field(self) -> None:
        value = submit_payload()
        value.pop("job_id")
        value.update(op="status", args={})
        request = ProviderRequest.from_payload(value)
        self.assertIsNone(request.job_id)
        value["job_id"] = None
        self.refusal(value, "INVALID_REQUEST")
        value = submit_payload()
        value.pop("schema")
        self.refusal(value, "INVALID_REQUEST")

    def test_descriptor_index_is_exact(self) -> None:
        value = submit_payload()
        value["args"]["audio_fd"] = 1
        self.refusal(value, "DESCRIPTOR_MISMATCH")

    def test_unhashable_audio_media_type_is_stably_refused(self) -> None:
        for malformed in ([], {}):
            with self.subTest(malformed=malformed):
                value = submit_payload()
                value["args"]["audio"]["media_type"] = malformed
                self.refusal(value, "UNSUPPORTED_CAPABILITY")

    def test_audio_descriptor_is_bound_to_digest_and_length(self) -> None:
        value = submit_payload()
        request = ProviderRequest.from_payload(value)
        with tempfile.TemporaryFile() as audio:
            audio.write(b"RIFF")
            audio.flush()
            snapshot = verify_request_descriptors(request, (audio.fileno(),))
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            try:
                self.assertNotEqual(snapshot, audio.fileno())
                self.assertEqual(
                    fcntl.fcntl(snapshot, fcntl.F_GETFL) & os.O_ACCMODE,
                    os.O_RDONLY,
                )
                audio.seek(0)
                audio.write(b"WAVE")
                audio.flush()
                self.assertEqual(os.pread(snapshot, 4, 0), b"RIFF")
            finally:
                os.close(snapshot)
            value["args"]["audio"]["sha256"] = "0" * 64
            with self.assertRaises(ProtocolError) as caught:
                verify_request_descriptors(ProviderRequest.from_payload(value),
                                           (audio.fileno(),))
            self.assertEqual(caught.exception.code, "DESCRIPTOR_MISMATCH")

    def test_non_submit_descriptor_is_refused(self) -> None:
        value = submit_payload()
        value.pop("job_id")
        value.update(op="status", args={})
        with tempfile.TemporaryFile() as audio:
            with self.assertRaises(ProtocolError) as caught:
                verify_request_descriptors(ProviderRequest.from_payload(value),
                                           (audio.fileno(),))
            self.assertEqual(caught.exception.code, "DESCRIPTOR_MISMATCH")


@unittest.skipUnless(hasattr(socket, "SOCK_SEQPACKET"), "SOCK_SEQPACKET unavailable")
class SocketTests(unittest.TestCase):
    def test_packet_and_descriptor_round_trip(self) -> None:
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        descriptor = os.open("/dev/null", os.O_RDONLY)
        try:
            send_packet(left, submit_payload(), descriptor)
            value, descriptors = receive_packet(right)
            self.assertEqual(value, submit_payload())
            self.assertEqual(len(descriptors), 1)
            os.fstat(descriptors[0])
            os.close(descriptors[0])
        finally:
            os.close(descriptor)
            left.close()
            right.close()

    def test_same_uid_peer_uses_kernel_credential(self) -> None:
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            self.assertEqual(require_same_uid_peer(left), os.geteuid())
            with self.assertRaises(ProtocolError) as caught:
                require_same_uid_peer(right, os.geteuid() + 1)
            self.assertEqual(caught.exception.code, "UNAUTHORIZED_PEER")
        finally:
            left.close()
            right.close()

    def test_transport_errors_are_stable(self) -> None:
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        left.close()
        try:
            for operation in (
                lambda: send_packet(left, submit_payload()),
                lambda: receive_packet(left),
                lambda: require_same_uid_peer(left),
            ):
                with self.assertRaises(ProtocolError) as caught:
                    operation()
                self.assertEqual(caught.exception.code, "TRANSPORT_ERROR")
        finally:
            right.close()


class AtomicOutputTests(unittest.TestCase):
    def transcript(self):
        assembler = TranscriptAssembler("transcribe")
        assembler.apply(SegmentUpdate(0, 0, 0, 500, "bounded result", True))
        return assembler.finish(engine_id="engine-not-selected", model_id="model-not-selected",
                                language="en")

    def test_commit_is_private_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            root = Path(parent) / "results"
            store = AtomicTranscriptStore(root)
            result = store.commit("job-1", self.transcript(), "json")
            self.assertEqual(result.parent, root)
            self.assertEqual(result.stat().st_mode & 0o777, 0o600)
            self.assertFalse(any(root.glob("*.partial")))
            self.assertEqual(json.loads(result.read_text())["segments"][0]["text"],
                             "bounded result")

    def test_caller_path_and_public_root_refuse(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            root = Path(parent) / "results"
            store = AtomicTranscriptStore(root)
            with self.assertRaisesRegex(Exception, "safe population"):
                store.commit("../escape", self.transcript(), "text")
            os.chmod(root, 0o755)
            with self.assertRaisesRegex(Exception, "not private"):
                AtomicTranscriptStore(root)

    def test_existing_result_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            store = AtomicTranscriptStore(Path(parent) / "results")
            result = store.commit("job-1", self.transcript(), "text")
            original = result.read_bytes()
            with self.assertRaises(Exception):
                store.commit("job-1", self.transcript(), "text")
            self.assertEqual(result.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
