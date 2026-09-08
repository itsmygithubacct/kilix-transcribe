"""Synthetic recorder helpers and real provider sockets; no physical capture."""
from contextlib import contextmanager
import io
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import wave

from kilix_transcribe.protocol import ProtocolError
from kilix_transcribe.recording import capture_wav, record_and_transcribe, receiver_closed
from kilix_transcribe.service import client_request, request_value

try:
    from voicelib.audio import MicCapture
    from voicelib.microphone import CaptureControl
except ImportError:
    MicCapture = CaptureControl = None


@contextmanager
def provider(mode="success"):
    from test_runtime import RuntimeTests
    fixture = RuntimeTests()
    fixture.setUp()
    try:
        fixture.start(mode)
        yield fixture
    finally:
        fixture.doCleanups()


@unittest.skipIf(MicCapture is None, "managed voice microphone API is not installed")
class RecordingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="kt-record-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.phases = []
        self.before_fds = len(os.listdir("/proc/self/fd"))

    def tearDown(self):
        self.assertEqual(len(os.listdir("/proc/self/fd")), self.before_fds)

    def options(self, source="import os;os.write(1,b'\\x01\\0'*6400)", **changes):
        value = dict(seconds=1, indicator=self.phases.append, locked=lambda: False,
                     namespace=str(self.root / "microphone"),
                     config={"audio": {"capture_cmd": [sys.executable, "-B", "-c", source]}})
        value.update(changes)
        return value

    def test_actual_pcm_and_only_proved_inactive_after_source_end(self):
        clip = capture_wav(**self.options())
        with wave.open(io.BytesIO(clip.wav), "rb") as audio:
            self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()), (1, 2, 16000))
            self.assertEqual(audio.readframes(audio.getnframes()), b"\x01\0" * 6400)
        self.assertEqual(clip.duration_ms, 400)
        self.assertEqual(self.phases[0], "recording")
        self.assertEqual(self.phases[-1], "inactive")

    def test_actual_controls_stop_without_returning_audio(self):
        source = "import os,time\nwhile True:os.write(1,b'\\x01\\0'*320);time.sleep(.02)"
        for kind in ("cancelled", "disconnected", "locked", "provider_alive"):
            with self.subTest(kind=kind):
                event = threading.Event()
                timer = threading.Timer(.15, event.set)
                timer.start()
                options = self.options(source, **{kind: (lambda: not event.is_set()) if kind == "provider_alive" else event.is_set})
                try:
                    with self.assertRaises(ProtocolError):
                        capture_wav(**options)
                finally:
                    timer.cancel()
                    timer.join(1)
                self.assertEqual(self.phases[-1], "inactive")

    def test_energy_vad_stops_after_speech_and_keeps_original_pcm(self):
        source = "import os;os.write(1,b'\\xff\\x1f'*320*30+b'\\0\\0'*320*60)"
        clip = capture_wav(**self.options(source, seconds=3, vad=True))
        self.assertTrue(clip.speech_detected)
        self.assertEqual(clip.ended_by, "silence")
        self.assertEqual(clip.duration_ms, 1500)
        self.assertEqual(clip.wav[44:], b"\xff\x1f" * 320 * 30 + b"\0\0" * 320 * 45)

    def test_real_duration_limit_stops_live_helper_before_return(self):
        marker = self.root / "recorder.pid"
        source = (f"import os,time;from pathlib import Path;Path({str(marker)!r}).write_text(str(os.getpid()))\n"
                  "while True:os.write(1,b'\\1\\0'*320);time.sleep(.02)")
        started = time.monotonic()
        clip = capture_wav(**self.options(source))
        self.assertGreaterEqual(time.monotonic() - started, .95)
        self.assertLess(time.monotonic() - started, 3)
        self.assertTrue(0 < clip.duration_ms <= 1000)
        self.assertFalse((Path("/proc") / marker.read_text()).exists())
        self.assertEqual(self.phases[-1], "inactive")

    def test_invalid_control_and_duration_never_launch_recorder(self):
        marker = self.root / "opened"
        source = f"from pathlib import Path;Path({str(marker)!r}).touch()"
        for value in (True, None, "1", 0, 121, float("nan"), float("inf"), 10**10000):
            with self.subTest(kind=type(value).__name__), self.assertRaises(ProtocolError):
                capture_wav(**self.options(source, seconds=value))
        for name, callback in (("locked", lambda: None), ("cancelled", lambda: True),
                               ("disconnected", lambda: True), ("provider_alive", lambda: False)):
            with self.subTest(control=name), self.assertRaises(ProtocolError):
                capture_wav(**self.options(source, **{name: callback}))
        self.assertFalse(marker.exists())
        for config in ({"audio": None}, {"audio": []}, {"audio": {"rate": 1}}, {"vad": {}}):
            with self.subTest(config=config), self.assertRaises(ProtocolError):
                capture_wav(**self.options(config=config))

    def test_indicator_failure_refuses_before_open(self):
        marker = self.root / "opened"
        def broken(_phase):
            raise OSError("synthetic unavailable indicator")
        with self.assertRaises(ProtocolError):
            capture_wav(**self.options(f"from pathlib import Path;Path({str(marker)!r}).touch()", indicator=broken))
        self.assertFalse(marker.exists())

    def test_real_provider_receives_audio_only_after_capture_cleanup(self):
        with provider() as fixture:
            import kilix_transcribe.service as module
            original = module.run_job
            def checked(*args, **kwargs):
                self.assertEqual(self.phases[-1], "inactive")
                return original(*args, **kwargs)
            with patch.object(module, "run_job", side_effect=checked):
                result = record_and_transcribe(fixture.ipc, **self.options(), timeout=5)
            self.assertEqual(result["text"], "Test transcript.\n")
            self.assertEqual(result["duration_ms"], 400)
            self.assertEqual(client_request(fixture.ipc, request_value("status"))["provider_state"], "ready")

    def test_unknown_capability_and_unproved_capture_never_submit(self):
        with provider() as fixture:
            with self.assertRaises(ProtocolError) as error:
                record_and_transcribe(fixture.ipc, **self.options(), task="diarize", timeout=5)
            self.assertEqual(error.exception.code, "UNSUPPORTED_CAPABILITY")
            self.assertEqual(self.phases, [])
            original_stop = MicCapture.stop
            def unproved(capture):
                original_stop(capture)
                return False
            with patch.object(MicCapture, "stop", unproved), self.assertRaises(ProtocolError) as error:
                record_and_transcribe(fixture.ipc, **self.options(), timeout=5)
            self.assertEqual(error.exception.code, "SUPERVISOR_FAILED")
            self.assertFalse((fixture.installation / "engine.pid").exists())

    def test_dropped_frames_never_submit(self):
        with provider() as fixture:
            # Force a real recorder to fill/drop from the bounded queue before
            # the consumer begins, using a file written only after all PCM.
            marker = self.root / "all-written"
            source = f"import os;from pathlib import Path;os.write(1,b'\\1\\0'*320*800);Path({str(marker)!r}).touch()"
            original = MicCapture.start
            def start(capture):
                original(capture)
                until = time.monotonic() + 2
                while not marker.exists() and time.monotonic() < until:
                    time.sleep(.005)
                self.assertTrue(marker.exists())
                self.assertGreater(capture.overruns, 0)
            with patch.object(MicCapture, "start", start), self.assertRaises(ProtocolError):
                record_and_transcribe(fixture.ipc, **self.options(source), timeout=5)
            self.assertFalse((fixture.installation / "engine.pid").exists())

    def test_reader_allocation_failure_still_stops_real_recorder(self):
        with provider() as fixture:
            original = threading.Thread
            marker = self.root / "reader-failure.pid"
            source = (f"import os,time;from pathlib import Path;Path({str(marker)!r}).write_text(str(os.getpid()))\n"
                      "while True:os.write(1,b'\\1\\0'*320);time.sleep(.02)")
            def allocate(*args, **kwargs):
                if kwargs.get("name") == "kilix-voice-mic":
                    until = time.monotonic() + 2
                    while not marker.exists() and time.monotonic() < until:
                        time.sleep(.005)
                    self.assertTrue(marker.exists())
                    raise RuntimeError("synthetic reader allocation failure")
                return original(*args, **kwargs)
            with patch.object(threading, "Thread", side_effect=allocate), self.assertRaises(ProtocolError):
                record_and_transcribe(fixture.ipc, **self.options(source), timeout=5)
            self.assertFalse((Path("/proc") / marker.read_text()).exists())
            self.assertFalse((fixture.installation / "engine.pid").exists())
            self.assertEqual(self.phases[-1], "inactive")

    def test_cancel_during_asr_waits_for_actual_escaped_tree_reap(self):
        with provider("escape") as fixture:
            cancellation = threading.Event()
            finished = threading.Event()
            def wait_for_engine():
                until = time.monotonic() + 3
                while time.monotonic() < until and not finished.is_set():
                    if (fixture.installation / "escaped.pid").exists():
                        cancellation.set()
                        return
                    time.sleep(.01)
            observer = threading.Thread(target=wait_for_engine)
            observer.start()
            try:
                with self.assertRaises(ProtocolError) as error:
                    record_and_transcribe(fixture.ipc, **self.options(cancelled=cancellation.is_set), timeout=8)
                self.assertEqual(error.exception.code, "CANCELED")
                for name in ("engine.pid", "escaped.pid"):
                    self.assertFalse((Path("/proc") / (fixture.installation / name).read_text()).exists())
                self.assertEqual(client_request(fixture.ipc, request_value("status"))["provider_state"], "ready")
            finally:
                finished.set()
                observer.join(4)
            self.assertFalse(observer.is_alive())

    def test_actual_provider_loss_stops_capture_without_submission(self):
        with provider() as fixture:
            timer = threading.Timer(.1, fixture.service.stop)
            timer.start()
            source = "import os,time\nwhile True:os.write(1,b'\\1\\0'*320);time.sleep(.02)"
            try:
                with self.assertRaises(ProtocolError):
                    record_and_transcribe(fixture.ipc, **self.options(source), timeout=5)
            finally:
                timer.cancel()
                timer.join(1)
            self.assertFalse((fixture.installation / "engine.pid").exists())
            self.assertEqual(self.phases[-1], "inactive")


class ClientTests(unittest.TestCase):
    def test_file_fifo_refuses_without_open_wait(self):
        with tempfile.TemporaryDirectory(prefix="kt-fifo-") as directory:
            fifo = Path(directory) / "input.wav"
            os.mkfifo(fifo)
            result = subprocess.run([sys.executable, "-B", "-m", "kilix_transcribe", "file", str(fifo)],
                                    capture_output=True, timeout=2)
            self.assertEqual(result.returncode, 69)
            self.assertIn(b"INVALID_REQUEST", result.stderr)

    def test_already_canceled_or_invalid_callback_never_submits(self):
        with provider() as fixture, fixture.audio.open("rb") as audio:
            for callback, expected in ((lambda: True, "CANCELED"), (lambda: [], "INVALID_REQUEST")):
                with self.assertRaises(ProtocolError) as error:
                    client_request(fixture.ipc, fixture.value(), audio.fileno(), cancelled=callback)
                self.assertEqual(error.exception.code, expected)
                self.assertFalse((fixture.installation / "engine.pid").exists())

    def test_receiver_pipe_lifetime_without_writing_text(self):
        reader, writer = os.pipe2(os.O_CLOEXEC)
        try:
            self.assertFalse(receiver_closed(writer))
            os.close(reader)
            reader = -1
            self.assertTrue(receiver_closed(writer))
        finally:
            if reader >= 0:
                os.close(reader)
            os.close(writer)

    def test_terminal_operation_binding_uses_real_provider_socket(self):
        import kilix_transcribe.service as module
        with provider() as fixture:
            original = module.send_packet
            for operation, replacement in (("models", "status"), ("status", "models"),
                                           ("unload", "canceled"), ("cancel", "unloaded"),
                                           ("submit", "status")):
                def wrong(channel, value, descriptor=None):
                    if value["type"] in {"models", "status", "unloaded", "canceled", "result"}:
                        value = dict(value, type=replacement, result={})
                        descriptor = None
                    return original(channel, value, descriptor)
                with self.subTest(operation=operation), patch.object(module, "send_packet", side_effect=wrong):
                    request = fixture.value() if operation == "submit" else request_value(
                        operation, job_id="job" if operation == "cancel" else None)
                    with fixture.audio.open("rb") as audio, self.assertRaises(ProtocolError) as error:
                        client_request(fixture.ipc, request, audio.fileno() if operation == "submit" else None)
                    self.assertEqual(error.exception.code, "INVALID_RESPONSE")
