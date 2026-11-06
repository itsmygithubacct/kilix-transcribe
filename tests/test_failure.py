"""Worker-stop classification without a device, a model, or an engine."""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from kilix_transcribe.failure import CorruptInput, classify, guard, report
from kilix_transcribe.supervisor import exit_status


class FailureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="kt-failure-")
        self.addCleanup(self.temporary.cleanup)
        self.workspace = self.temporary.name

    def test_exit_status_preserves_stop_and_signal_death(self):
        self.assertEqual(exit_status(-9, False), 137)
        self.assertEqual(exit_status(-15, False), 143)
        self.assertEqual(exit_status(0, False), 0)
        self.assertEqual(exit_status(1, False), 1)
        self.assertEqual(exit_status(3, False), 3)
        self.assertEqual(exit_status(None, True), 130)
        self.assertEqual(exit_status(-9, True), 130)
        self.assertEqual(exit_status(None, False), 125)

    def test_signals_and_disk_exit_do_not_need_a_record(self):
        for code in (-9, -15, 132, 134, 135, 136, 137, 139, 143, 159):
            with self.subTest(code=code):
                self.assertEqual(classify(code, self.workspace)[0], "WORKER_CRASH")
        self.assertIsNone(classify(130, self.workspace))
        self.assertIsNone(classify(1, self.workspace))
        self.assertIsNone(classify(0, self.workspace))
        self.assertEqual(classify(4, None),
                         ("INSUFFICIENT_DISK", "speech worker ran out of disk"))

    def test_only_a_small_regular_record_selects_a_code(self):
        report_path = Path(self.workspace) / "failure.json"
        for kind, code in (
            ("oom", "DEVICE_UNAVAILABLE"),
            ("device_loss", "DEVICE_UNAVAILABLE"),
            ("corrupt", "INVALID_REQUEST"),
            ("disk", "INSUFFICIENT_DISK"),
        ):
            with self.subTest(kind=kind):
                report_path.write_text(json.dumps({"kind": kind}, separators=(",", ":")))
                self.assertEqual(classify(3, self.workspace)[0], code)
        report_path.write_text('{"kind":"oom"}')
        self.assertIsNone(classify(1, self.workspace))
        report_path.write_text('{"kind":"oom","extra":1}')
        self.assertIsNone(classify(3, self.workspace))
        report_path.write_text("not-json")
        self.assertIsNone(classify(3, self.workspace))
        report_path.write_text('{"kind":"oom"}' + (" " * 64))
        self.assertIsNone(classify(3, self.workspace))
        report_path.unlink()
        report_path.symlink_to(self.workspace)
        self.assertIsNone(classify(3, self.workspace))
        report_path.unlink()
        os.mkfifo(report_path)
        started = time.monotonic()
        self.assertIsNone(classify(3, self.workspace))
        self.assertLess(time.monotonic() - started, 1)
        os.remove(report_path)
        report_path.write_text('{"kind":"oom"}')
        real = os.fstat

        def wrong_owner(descriptor):
            info = real(descriptor)
            return type("Stat", (), {
                "st_mode": info.st_mode, "st_uid": info.st_uid + 1, "st_size": info.st_size,
            })()

        with patch("kilix_transcribe.failure.os.fstat", side_effect=wrong_owner):
            self.assertIsNone(classify(3, self.workspace))
        self.assertEqual(classify(137, self.workspace)[0], "WORKER_CRASH")
        self.assertEqual(classify(4, self.workspace)[0], "INSUFFICIENT_DISK")

    def test_report_exits_and_a_full_disk_does_not_recurse(self):
        with self.assertRaises(SystemExit) as caught:
            report(self.workspace, "disk")
        self.assertEqual(caught.exception.code, 3)
        self.assertEqual(json.loads(Path(self.workspace, "failure.json").read_text()),
                         {"kind": "disk"})
        self.assertEqual(Path(self.workspace, "failure.json").stat().st_mode & 0o777, 0o600)

        def full(*_args, **_kwargs):
            raise OSError(errno.ENOSPC, "no space")

        with patch("kilix_transcribe.failure.os.open", side_effect=full):
            with self.assertRaises(SystemExit) as caught:
                report(self.workspace + "-missing", "disk")
            self.assertEqual(caught.exception.code, 4)
            with self.assertRaises(OSError):
                report(self.workspace + "-missing", "corrupt")

    def test_guard_records_memory_disk_and_corrupt_input(self):
        def stop(error):
            raise error

        with self.assertRaises(SystemExit) as caught:
            guard(self.workspace, lambda: stop(MemoryError()))
        self.assertEqual(caught.exception.code, 3)
        self.assertEqual(json.loads(Path(self.workspace, "failure.json").read_text())["kind"], "oom")
        os.remove(Path(self.workspace, "failure.json"))
        with self.assertRaises(SystemExit):
            guard(self.workspace, lambda: stop(OSError(errno.EDQUOT, "quota")))
        self.assertEqual(json.loads(Path(self.workspace, "failure.json").read_text())["kind"], "disk")
        os.remove(Path(self.workspace, "failure.json"))
        with self.assertRaises(SystemExit):
            guard(self.workspace, lambda: stop(CorruptInput("bad")))
        self.assertEqual(json.loads(Path(self.workspace, "failure.json").read_text())["kind"], "corrupt")
        os.remove(Path(self.workspace, "failure.json"))
        with self.assertRaises(OSError):
            guard(self.workspace, lambda: stop(OSError(errno.ENOENT, "missing")))
        self.assertFalse((Path(self.workspace) / "failure.json").exists())
        with self.assertRaises(MemoryError):
            guard(None, lambda: stop(MemoryError()))


if __name__ == "__main__":
    unittest.main()
