"""Installed startup and staging use the real receipt and Content authority."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from kilix_transcribe.cli import main
from kilix_transcribe.protocol import ProtocolError
from kilix_transcribe.runtime import ENGINE_COMMIT, RUNTIME_SCHEMA
import test_content as fixtures


REVISION = "5359861c739e955e79d9a303bcbc70fb988958b1"
PAYLOAD = b"synthetic installed model"


@unittest.skipUnless(fixtures.CONTENT_AVAILABLE, "reviewed content installed API required")
class StartupTests(unittest.TestCase):
    def fixture(self, *, receipt=True):
        installed = self.enterContext(fixtures.PackagedFixture(
            {"model.bin": PAYLOAD}, asset_id="whisper-tiny-ggml",
            revision=REVISION, receipt=receipt))
        root = installed.root / "runtime"
        root.mkdir(mode=0o700)
        files = {"model.bin": hashlib.sha256(PAYLOAD).hexdigest()}
        for name in ("whisper-cli", "ffmpeg"):
            payload = b"#!/bin/sh\nexit 0\n"
            (root / name).write_bytes(payload)
            (root / name).chmod(0o700)
            files[name] = hashlib.sha256(payload).hexdigest()
        (root / "runtime.json").write_text(json.dumps({
            "schema": RUNTIME_SCHEMA, "engine_revision": ENGINE_COMMIT,
            "model": {"id": "whisper-tiny", "revision": REVISION}, "files": files}))
        return installed, root, files

    def serve(self, installed, root):
        return main(["serve", "--runtime-root", str(root),
            "--installed-asset", "whisper-tiny-ggml", "--content-root", str(installed.data),
            "--model-snapshot-bytes", "1024"])

    def test_missing_receipt_refuses_before_service_creation(self):
        installed, root, _files = self.fixture(receipt=False)
        with patch("kilix_transcribe.service.Service") as service, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.serve(installed, root), 69)
        service.assert_not_called()

    def test_corrupt_model_refuses_before_service_creation(self):
        installed, root, _files = self.fixture()
        (installed.selected / "model.bin").write_bytes(b"broken")
        with patch("kilix_transcribe.service.Service") as service, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.serve(installed, root), 69)
        service.assert_not_called()

    def test_valid_installed_model_starts_without_local_copy(self):
        installed, root, _files = self.fixture()
        service = Mock()
        with patch("kilix_transcribe.service.Service", return_value=service), patch("signal.signal"):
            self.assertEqual(self.serve(installed, root), 0)
        service.serve.assert_called_once_with()
        self.assertFalse((root / "model.bin").exists())

    def stage(self, installed, root, files, destination):
        path = Path(__file__).resolve().parents[1] / "tools/stage_runtime.py"
        spec = importlib.util.spec_from_file_location("_stage_runtime_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        arguments = [str(path), "--destination", str(destination),
            "--engine", str(root / "whisper-cli"), "--engine-sha256", files["whisper-cli"],
            "--decoder", str(root / "ffmpeg"), "--decoder-sha256", files["ffmpeg"],
            "--model-sha256", files["model.bin"], "--model-id", "whisper-tiny",
            "--model-revision", REVISION, "--installed-asset", "whisper-tiny-ggml",
            "--content-root", str(installed.data), "--model-snapshot-bytes", "1024"]
        with patch("sys.argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            module.main()

    def test_installed_staging_has_only_tools_and_manifest(self):
        installed, root, files = self.fixture()
        destination = installed.root / "staged"
        self.stage(installed, root, files, destination)
        self.assertEqual({p.name for p in destination.iterdir()}, {"runtime.json", "whisper-cli", "ffmpeg"})
        self.assertEqual(json.loads((destination / "runtime.json").read_text())["files"], files)

    def test_missing_receipt_prevents_staging_publication(self):
        installed, root, files = self.fixture(receipt=False)
        destination = installed.root / "staged"
        with self.assertRaises(ProtocolError):
            self.stage(installed, root, files, destination)
        self.assertFalse(destination.exists())

    def test_corrupt_installed_model_prevents_staging_publication(self):
        installed, root, files = self.fixture()
        (installed.selected / "model.bin").write_bytes(b"broken")
        destination = installed.root / "staged"
        with self.assertRaises(ProtocolError):
            self.stage(installed, root, files, destination)
        self.assertFalse(destination.exists())
