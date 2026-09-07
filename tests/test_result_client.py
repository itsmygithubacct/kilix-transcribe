"""Malformed same-UID provider results must remain bounded refusals."""
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import unittest

from kilix_transcribe.protocol import PROTOCOL_SCHEMA, ProtocolError, receive_packet, send_packet
from kilix_transcribe.runtime import ENGINE_COMMIT
from kilix_transcribe.service import client_request, request_value
from kilix_transcribe.surface import Transcript, render_transcript


class ResultClientTests(unittest.TestCase):
    def document(self):
        transcript = Transcript("transcribe", "whisper.cpp", "test-model", "en", ())
        result = json.loads(render_transcript(transcript, "json"))
        result.update(engine_revision=ENGINE_COMMIT, model_revision="test-revision",
                      duration_ms=1000, text="", output_format="text", output="")
        return result

    def receive(self, payload, *, writable=False, mutate=lambda value: None):
        with tempfile.TemporaryDirectory(prefix="kt-result-") as directory:
            root = Path(directory)
            with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as listener:
                listener.bind(str(root / "kilix-transcribe.sock"))
                os.chmod(root / "kilix-transcribe.sock", 0o600)
                listener.listen(1)
                errors = []
                def serve():
                    try:
                        channel, _ = listener.accept()
                        with channel:
                            request, descriptors = receive_packet(channel)
                            self.assertFalse(descriptors)
                            with tempfile.TemporaryFile() as writer:
                                writer.write(payload); writer.flush()
                                fd = os.open(f"/proc/self/fd/{writer.fileno()}",
                                             (os.O_RDWR if writable else os.O_RDONLY) | os.O_CLOEXEC)
                                try:
                                    result = {"transcript": {"fd": 0, "byte_length": len(payload),
                                                              "sha256": hashlib.sha256(payload).hexdigest()},
                                              "engine_id": "whisper.cpp", "engine_revision": ENGINE_COMMIT,
                                              "model_id": "test-model", "model_revision": "test-revision",
                                              "duration_ms": 1000}
                                    event = {"schema": PROTOCOL_SCHEMA, "type": "result",
                                             "request_id": request["request_id"], "job_id": request["job_id"],
                                             "result": result}
                                    mutate(event)
                                    send_packet(channel, event, fd)
                                finally:
                                    os.close(fd)
                    except Exception as error:
                        errors.append(error)
                thread = threading.Thread(target=serve)
                thread.start()
                try:
                    return client_request(root, request_value("submit", job_id="result-test", timeout=2,
                        args={"task": "transcribe", "language": "en", "output": "text", "audio_fd": 0,
                              "audio": {"media_type": "audio/wav", "byte_length": 1, "sha256": "0" * 64}}))
                finally:
                    thread.join(3)
                    self.assertFalse(thread.is_alive())
                    self.assertEqual(errors, [])

    def test_valid_empty_transcript(self):
        result = self.document()
        self.assertEqual(self.receive(json.dumps(result).encode()), result)

    def test_invalid_json_population_and_descriptor_mode_refuse(self):
        invalid = (b'{"output":"first","output":"second"}', b'{"output":NaN}', b'[]',
                   b'{"output":' + b'[' * 100 + b'0' + b']' * 100 + b'}')
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ProtocolError):
                self.receive(payload)
        with self.assertRaises(ProtocolError):
            self.receive(json.dumps(self.document()).encode(), writable=True)

    def test_malformed_envelope_and_unbound_document_refuse(self):
        for mutation in (lambda e: e.update(result=[]),
                         lambda e: e["result"]["transcript"].update(fd=False),
                         lambda e: e["result"].update(model_id="different"),
                         lambda e: e["result"].update(duration_ms=True)):
            with self.subTest(mutation=mutation), self.assertRaises(ProtocolError):
                self.receive(json.dumps(self.document()).encode(), mutate=mutation)
        for field, value in (("segments", [None]), ("output_format", []), ("duration_ms", True),
                             ("output", "not the segment serialization")):
            with self.subTest(field=field), self.assertRaises(ProtocolError):
                document = self.document(); document[field] = value
                self.receive(json.dumps(document).encode())
