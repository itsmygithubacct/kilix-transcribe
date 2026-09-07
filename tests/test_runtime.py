"""Supervision/IPC tests using explicit fake tools, not model qualification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import wave

from kilix_transcribe.protocol import ProtocolError, receive_packet, send_packet
from kilix_transcribe.runtime import ENGINE_COMMIT, InstalledRuntime, digest_file
from kilix_transcribe.service import Service, client_request, request_value


DECODER = """#!/usr/bin/python3
import shutil, sys
shutil.copyfile(sys.argv[sys.argv.index('-i')+1],sys.argv[-1])
"""
ENGINE = """#!/usr/bin/python3
import json, os, pathlib, signal, subprocess, sys, time
root=pathlib.Path(sys.argv[0]).parent
mode=(root/'model.bin').read_text()
(root/'engine.pid').write_text(str(os.getpid()))
if mode=='sleep':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(120)
if mode=='fail':
    raise SystemExit(3)
destination=pathlib.Path(sys.argv[sys.argv.index('--output-file')+1]+'.json')
destination.write_text(json.dumps({'result':{'language':'en'},'transcription':[
 {'offsets':{'from':0,'to':1000},'text':'Test transcript.',
  'tokens':[{'text':' Test','offsets':{'from':0,'to':400}},
            {'text':' transcript.','offsets':{'from':400,'to':1000}}]}]}))
"""


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="kt-runtime-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.installation = self.root / "installation"
        self.installation.mkdir(mode=0o700)
        self.ipc = self.root / "ipc"
        self.ipc.mkdir(mode=0o700)
        self.audio = self.root / "speech.wav"
        with wave.open(str(self.audio), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"\x01\x00" * 16000)
        self.service = None
        self.thread = None
        self.errors = []
        self.addCleanup(self.stop)

    def runtime(self, mode="success"):
        for name, source in (("ffmpeg", DECODER), ("whisper-cli", ENGINE), ("model.bin", mode)):
            path = self.installation / name
            path.write_text(source)
            path.chmod(0o600 if name == "model.bin" else 0o700)
        manifest = {"schema": "kilix.transcribe.runtime/v1", "engine_revision": ENGINE_COMMIT,
                    "model": {"id": "test-model", "revision": "test-revision"},
                    "files": {name: digest_file(self.installation / name)
                              for name in ("ffmpeg", "whisper-cli", "model.bin")}}
        (self.installation / "runtime.json").write_text(json.dumps(manifest))
        return InstalledRuntime(self.installation)

    def test_staging_verifies_bytes_and_preserves_existing_destination(self):
        self.runtime()
        destination = self.root / "staged"
        tool = Path(__file__).resolve().parents[1] / "tools" / "stage_runtime.py"
        command = [sys.executable, str(tool), "--destination", str(destination),
                   "--model-id", "test-model", "--model-revision", "test-revision"]
        for argument, name in (("engine", "whisper-cli"), ("decoder", "ffmpeg"), ("model", "model.bin")):
            command.extend([f"--{argument}", str(self.installation / name),
                            f"--{argument}-sha256", digest_file(self.installation / name)])
        invalid = command.copy()
        invalid[-1] = "0" * 64
        self.assertNotEqual(subprocess.run(invalid, capture_output=True).returncode, 0)
        self.assertFalse(destination.exists())
        staged = subprocess.run(command, capture_output=True)
        self.assertEqual(staged.returncode, 0, staged.stderr)
        self.assertEqual(InstalledRuntime(destination).model_id, "test-model")
        marker = destination / "keep"
        marker.write_text("existing runtime")
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual(marker.read_text(), "existing runtime")

    def start(self, mode="success"):
        self.service = Service(self.runtime(mode), self.ipc)
        def target():
            try:
                self.service.serve()
            except Exception as error:
                self.errors.append(error)
        self.thread = threading.Thread(target=target)
        self.thread.start()
        self.assertTrue(self.service.ready.wait(3), self.errors)

    def stop(self):
        if self.service is not None:
            self.service.stop()
        if self.thread is not None:
            self.thread.join(5)
            self.assertFalse(self.thread.is_alive(), "provider did not stop")
        self.assertEqual(self.errors, [])

    def value(self, job="test-job", timeout=5, output="json", task="transcribe"):
        payload = self.audio.read_bytes()
        return request_value("submit", job_id=job, timeout=timeout, args={
            "task": task, "language": "en", "output": output, "audio_fd": 0,
            "audio": {"media_type": "audio/wav", "byte_length": len(payload),
                      "sha256": hashlib.sha256(payload).hexdigest()}})

    def submit(self, **kwargs):
        with self.audio.open("rb") as source:
            return client_request(self.ipc, self.value(**kwargs), source.fileno())

    def raw_submit(self, **kwargs):
        channel = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.addCleanup(channel.close)
        channel.settimeout(5)
        channel.connect(str(self.service.path))
        with self.audio.open("rb") as source:
            send_packet(channel, self.value(**kwargs), source.fileno())
        accepted, descriptors = receive_packet(channel)
        self.assertEqual(accepted["type"], "accepted")
        self.assertEqual(descriptors, ())
        return channel

    def await_worker(self):
        marker = self.installation / "engine.pid"
        deadline = time.monotonic() + 3
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(marker.exists(), "engine did not start")
        return int(marker.read_text())

    def test_runtime_digest_and_later_mutation_refuse(self):
        runtime = self.runtime()
        (self.installation / "model.bin").write_text("changed")
        with self.assertRaises(ProtocolError):
            runtime.verify_unchanged()
        with self.assertRaises(ProtocolError):
            InstalledRuntime(self.installation)

    def test_runtime_symlink_and_public_root_refuse(self):
        self.runtime()
        model = self.installation / "model.bin"
        model.unlink()
        model.symlink_to(self.audio)
        with self.assertRaises(ProtocolError):
            InstalledRuntime(self.installation)
        self.installation.chmod(0o755)
        with self.assertRaises(ProtocolError):
            InstalledRuntime(self.installation)

    def test_live_socket_cannot_be_replaced(self):
        self.start()
        identity = self.service.path.stat().st_ino
        other = Service(self.service.runtime, self.ipc)
        with self.assertRaises(ProtocolError) as raised:
            other.serve()
        self.assertEqual(raised.exception.code, "BUSY")
        self.assertEqual(self.service.path.stat().st_ino, identity)
        result = client_request(self.ipc, request_value("status"))
        self.assertEqual(result["provider_state"], "ready")

    def test_private_socket_and_all_outputs_from_worker(self):
        self.start()
        self.assertEqual(self.service.path.stat().st_mode & 0o777, 0o600)
        for output in ("text", "json", "webvtt", "srt"):
            result = self.submit(output=output)
            self.assertEqual(result["text"], "Test transcript.\n")
            self.assertEqual(result["engine_revision"], ENGINE_COMMIT)
            self.assertEqual(result["output_format"], output)
            self.assertEqual(len(result["segments"][0]["words"]), 2)

    def test_descriptor_digest_mismatch_refuses_before_engine(self):
        self.start()
        value = self.value()
        value["args"]["audio"]["sha256"] = "0" * 64
        with self.audio.open("rb") as source, self.assertRaises(ProtocolError) as raised:
            client_request(self.ipc, value, source.fileno())
        self.assertEqual(raised.exception.code, "DESCRIPTOR_MISMATCH")
        self.assertFalse((self.installation / "engine.pid").exists())

    def test_failed_worker_does_not_kill_service(self):
        self.start("fail")
        with self.assertRaises(ProtocolError) as raised:
            self.submit()
        self.assertEqual(raised.exception.code, "ENGINE_FAILED")
        self.assertEqual(client_request(self.ipc, request_value("status"))["provider_state"], "ready")

    def test_cancel_reaps_a_worker_ignoring_termination(self):
        self.start("sleep")
        channel = self.raw_submit()
        pid = self.await_worker()
        started = time.monotonic()
        result = client_request(self.ipc, request_value("cancel", job_id="test-job"))
        self.assertTrue(result["cancel_requested"])
        event, descriptors = receive_packet(channel)
        self.assertEqual(descriptors, ())
        self.assertEqual(event["error"]["code"], "CANCELED")
        self.assertLess(time.monotonic() - started, 2)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_deadline_reaps_worker(self):
        self.start("sleep")
        channel = self.raw_submit(timeout=0.8)
        pid = self.await_worker()
        event, descriptors = receive_packet(channel)
        self.assertEqual(descriptors, ())
        self.assertEqual(event["error"]["code"], "DEADLINE_EXCEEDED")
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_disconnect_cancels_and_releases_busy_slot(self):
        self.start("sleep")
        channel = self.raw_submit()
        pid = self.await_worker()
        with self.assertRaises(ProtocolError) as raised:
            self.submit(job="second")
        self.assertEqual(raised.exception.code, "BUSY")
        channel.close()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = client_request(self.ipc, request_value("status"))
            if status["provider_state"] == "ready":
                break
            time.sleep(0.01)
        self.assertEqual(status["provider_state"], "ready")
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_unavailable_diarization_is_an_explicit_refusal(self):
        self.start()
        with self.assertRaises(ProtocolError) as raised:
            self.submit(task="diarize")
        self.assertEqual(raised.exception.code, "UNSUPPORTED_CAPABILITY")

    def test_silence_returns_empty_without_starting_engine(self):
        self.start()
        with wave.open(str(self.audio), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(bytes(32000))
        result = self.submit()
        self.assertEqual(result["text"], "")
        self.assertFalse((self.installation / "engine.pid").exists())


if __name__ == "__main__":
    unittest.main()
