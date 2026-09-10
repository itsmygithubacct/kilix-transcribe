"""Installed model bytes through actual transcribe sockets and owned workers."""
from contextlib import contextmanager
import json
import os
import threading
import time
import unittest
from unittest.mock import patch

from kilix_transcribe import runtime as runtime_module
from kilix_transcribe.runtime import InstalledRuntime
from kilix_transcribe.protocol import ProtocolError
from kilix_transcribe.service import Service, client_request, request_value
import test_content as content_fixtures
import test_runtime as runtime_fixtures


@unittest.skipUnless(content_fixtures.CONTENT_AVAILABLE, "reviewed content installed API required")
class ContentRuntimeTests(unittest.TestCase):
    def setup_runtime(self, mode="success"):
        fixture = runtime_fixtures.RuntimeTests("runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        runtime = fixture.runtime(mode)
        manifest = runtime.manifest
        manifest["model"] = {"id": "whisper-tiny", "revision": "5359861c739e955e79d9a303bcbc70fb988958b1"}
        (fixture.installation / "runtime.json").write_text(json.dumps(manifest))
        installed = self.enterContext(content_fixtures.PackagedFixture({"model.bin": mode.encode()},
            asset_id="whisper-tiny-ggml", revision=manifest["model"]["revision"]))
        (fixture.installation / "model.bin").unlink()
        source = installed.source()
        runtime = InstalledRuntime(fixture.installation, model_source=source)
        fixture.service = Service(runtime, fixture.ipc)
        fixture.thread = threading.Thread(target=fixture.service.serve)
        fixture.thread.start()
        self.assertTrue(fixture.service.ready.wait(3))
        return fixture, installed, source, runtime

    def test_socket_worker_uses_installed_snapshot_after_path_replacement(self):
        fixture, installed, source, _runtime = self.setup_runtime()
        original = source.open
        @contextmanager
        def replace_after_open(check):
            with original(check) as asset:
                (installed.selected / "model.bin").write_bytes(b"fail")
                yield asset
        source.open = replace_after_open
        baseline = len(os.listdir("/proc/self/fd"))
        result = fixture.submit()
        self.assertIn("Test transcript", result["output"])
        self.assertFalse((fixture.installation / "model.bin").exists())
        with self.assertRaises(ProtocolError):
            fixture.submit(job="changed")
        fixture.stop()
        self.assertLessEqual(len(os.listdir("/proc/self/fd")), baseline)

    def test_installed_escape_cancel_and_recovery(self):
        fixture, _installed, _source, _runtime = self.setup_runtime("escape")
        errors = []
        def submit():
            try:
                fixture.submit(job="escaped")
            except ProtocolError as error:
                errors.append(error.code)
        thread = threading.Thread(target=submit)
        thread.start()
        try:
            marker = fixture.installation / "escaped.pid"
            deadline = time.monotonic() + 3
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(marker.exists())
            pid = int(marker.read_text())
            client_request(fixture.ipc, request_value("cancel", job_id="escaped"))
            thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, ["CANCELED"])
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
            self.assertEqual(client_request(fixture.ipc, request_value("status"))["provider_state"], "ready")
        finally:
            fixture.stop()
            thread.join(3)

    def test_bad_runtime_digest_refuses_before_worker_spawn(self):
        fixture, _installed, source, _runtime = self.setup_runtime()
        value = json.loads((fixture.installation / "runtime.json").read_text())
        value["files"]["model.bin"] = "0" * 64
        (fixture.installation / "runtime.json").write_text(json.dumps(value))
        with patch.object(runtime_module.subprocess, "Popen", side_effect=AssertionError("must not spawn")):
            with self.assertRaises(ProtocolError):
                InstalledRuntime(fixture.installation, model_source=source)

    def test_spawn_failure_releases_content_and_tool_descriptors(self):
        fixture, _installed, _source, runtime = self.setup_runtime()
        baseline = len(os.listdir("/proc/self/fd"))
        with fixture.audio.open("rb") as audio:
            with patch.object(runtime_module.subprocess, "Popen", side_effect=OSError("synthetic spawn failure")):
                with self.assertRaises(OSError):
                    runtime_module.run_job(runtime, audio.fileno(), fixture.value()["args"],
                        deadline=time.monotonic() + 3, cancel=threading.Event())
        self.assertEqual(len(os.listdir("/proc/self/fd")), baseline)
