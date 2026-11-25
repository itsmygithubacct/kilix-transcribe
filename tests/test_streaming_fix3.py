"""Bounded delivery, shared-descriptor and retry controls; inert pipes only."""
import ctypes
import errno
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from kilix_transcribe import diagnostic_writer as writer
from kilix_transcribe.streaming import SEND_TIMEOUT
from test_streaming import until
from test_streaming_fixes import alive, handle

LIBC = ctypes.CDLL(None, use_errno=True)


def pidfd(pid):
    result = LIBC.syscall(434, pid, 0)
    if result < 0:
        raise OSError(ctypes.get_errno(), "owned fixture pidfd")
    return result


def send(fd, sig):
    if select.select([fd], [], [], 0)[0]:
        return False
    if LIBC.syscall(424, fd, sig, 0, 0) < 0 and ctypes.get_errno() != errno.ESRCH:
        raise OSError(ctypes.get_errno(), "owned fixture signal")
    return True


def full_pipe(blocking):
    reader, target = os.pipe()
    os.set_blocking(target, False)
    size = fcntl.fcntl(target, fcntl.F_GETPIPE_SZ)
    assert os.write(target, b'x' * size) == size
    os.set_blocking(target, blocking)
    return reader, target, size


class RetrySemantics(unittest.TestCase):
    def test_partial_writes_eintr_and_readiness_race_keep_exact_bytes(self):
        data = ('日本語 e\u0301 🤖\n' * 90).encode()
        output = bytearray()
        actions = iter(['interrupt', 'blocked', 'race', 'blocked', 'partial'])
        waits = []

        def write(fd, view):
            self.assertEqual(fd, 77)
            action = next(actions, 'partial')
            if action == 'interrupt':
                raise InterruptedError(errno.EINTR, 'inert interruption')
            if action in ('blocked', 'race'):
                raise BlockingIOError(errno.EAGAIN, 'inert backpressure')
            count = min(7, len(view))
            output.extend(view[:count])
            return count

        def ready(read, write, error, timeout):
            waits.append(timeout)
            if len(waits) == 1:
                raise InterruptedError(errno.EINTR, 'inert readiness interruption')
            return [], write, []

        with patch.object(writer.os, 'write', write), patch.object(writer.select, 'select', ready):
            self.assertTrue(writer.write_deadline(77, data, time.monotonic() + SEND_TIMEOUT))
        self.assertEqual(bytes(output), data)
        self.assertEqual(output.decode(), data.decode())
        self.assertGreaterEqual(len(waits), 3)
        self.assertTrue(all(0 < wait <= SEND_TIMEOUT for wait in waits))

    def test_retries_use_remaining_absolute_budget(self):
        clock = [0.17]
        waits = []

        def ready(read, write, error, timeout):
            waits.append(timeout)
            clock[0] += 0.02
            return [], write, []

        with patch.object(writer.time, 'monotonic', lambda: clock[0]), \
                patch.object(writer.os, 'write', side_effect=BlockingIOError(errno.EAGAIN, 'race')), \
                patch.object(writer.select, 'select', ready):
            self.assertFalse(writer.write_deadline(77, b'preview', .2))
        self.assertEqual(len(waits), 2)
        self.assertAlmostEqual(waits[0], .03)
        self.assertAlmostEqual(waits[1], .01)

    def test_atomic_chunks_end_at_unicode_boundaries(self):
        chunks = []
        data = ('日本語 e\u0301 🤖\n' * 500).encode()

        def write(fd, view):
            chunks.append(bytes(view))
            return len(view)

        with patch.object(writer.os, 'write', write):
            self.assertTrue(writer.write_deadline(77, data, time.monotonic() + SEND_TIMEOUT))
        self.assertEqual(b''.join(chunks), data)
        self.assertTrue(all(0 < len(chunk) <= 512 for chunk in chunks))
        for chunk in chunks:
            chunk.decode('utf-8', errors='strict')

    def test_closed_unusable_and_zero_progress_refuse(self):
        for error in [BrokenPipeError(errno.EPIPE, 'closed'), OSError(errno.EBADF, 'unusable')]:
            with self.subTest(error=error), patch.object(writer.os, 'write', side_effect=error):
                self.assertFalse(writer.write_deadline(77, b'preview', time.monotonic() + SEND_TIMEOUT))
        with patch.object(writer.os, 'write', return_value=0):
            self.assertFalse(writer.write_deadline(77, b'preview', time.monotonic() + SEND_TIMEOUT))


class DescriptorOwnership(unittest.TestCase):
    def test_blocked_receiver_spawns_no_helper_and_preserves_shared_flags(self):
        from kilix_transcribe.cli import _stderr_write
        from kilix_transcribe.protocol import ProtocolError
        from types import SimpleNamespace
        reader, target, _ = full_pipe(True)
        duplicate = os.dup(target)
        flags = fcntl.fcntl(target, fcntl.F_GETFL)
        start = time.monotonic()
        try:
            with patch.object(sys, "stderr", SimpleNamespace(fileno=lambda: target)), \
                    patch.object(subprocess, "Popen", side_effect=AssertionError("unowned helper")):
                with self.assertRaises(ProtocolError) as error:
                    _stderr_write("preview\n", required=True)
            self.assertEqual(error.exception.code, "SLOW_CONSUMER")
            self.assertLess(time.monotonic() - start, SEND_TIMEOUT + .15)
            self.assertEqual([fcntl.fcntl(fd, fcntl.F_GETFL) for fd in (target, duplicate)],
                             [flags, flags])
        finally:
            for fd in (reader, target, duplicate):
                os.close(fd)

    def test_expired_deadline_refuses_before_output(self):
        reader, target = os.pipe()
        try:
            self.assertFalse(writer.deliver(target, "日本語\n".encode(), time.monotonic() - .01))
            os.set_blocking(reader, False)
            with self.assertRaises(BlockingIOError):
                os.read(reader, 1024)
        finally:
            os.close(reader)
            os.close(target)

    def test_unsupported_socket_refuses_without_mutating_flags(self):
        import socket
        left, right = socket.socketpair()
        flags = fcntl.fcntl(left.fileno(), fcntl.F_GETFL)
        try:
            self.assertFalse(writer.deliver(left.fileno(), b"preview", time.monotonic() + SEND_TIMEOUT))
            self.assertEqual(fcntl.fcntl(left.fileno(), fcntl.F_GETFL), flags)
        finally:
            left.close()
            right.close()


class SharedRecovery(unittest.TestCase):
    def test_two_nonblocking_writers_recover_without_changing_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader,target,size = full_pipe(False)
            duplicate = os.dup(target)
            before = fcntl.fcntl(target,fcntl.F_GETFL)
            processes=[];fds=[]
            try:
                for index in range(2):
                    message=f'writer{index} 日本語 e\u0301 🤖\n'
                    script=('from pathlib import Path\nimport time\n'
                            'import kilix_transcribe.cli as cli\n'
                            f'Path({str(root/f"ready-{index}")!r}).touch()\n'
                            f'while not Path({str(root/f"gate-{index}")!r}).exists():time.sleep(.001)\n'
                            f'Path({str(root/f"begin-{index}")!r}).write_text(str(time.monotonic()))\n'
                            f'cli._stderr_write({message!r},required=True)\n')
                    p=subprocess.Popen([sys.executable,'-B','-c',script],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=target)
                    processes.append(p);fds.append(pidfd(p.pid))
                until(lambda:all((root/f'ready-{i}').exists() for i in range(2)),timeout=3)
                for i in range(2):(root/f'gate-{i}').touch()
                until(lambda:all((root/f'begin-{i}').exists() for i in range(2)),timeout=3)
                begins=[float((root/f'begin-{i}').read_text()) for i in range(2)]
                while time.monotonic()<max(begins)+.08:time.sleep(.001)
                freed=time.monotonic()
                self.assertLess(freed-min(begins),SEND_TIMEOUT)
                self.assertEqual(os.read(reader,size),b'x'*size)
                outputs=[p.communicate(timeout=2)[0] for p in processes]
                self.assertEqual([p.returncode for p in processes],[0,0])
                self.assertEqual(outputs,[b'',b''])
                os.set_blocking(reader,False)
                delivered=os.read(reader,1024).decode('utf-8',errors='strict')
                self.assertCountEqual(delivered.splitlines(),['writer0 日本語 e\u0301 🤖','writer1 日本語 e\u0301 🤖'])
                self.assertEqual([fcntl.fcntl(fd,fcntl.F_GETFL) for fd in (target,duplicate)],[before,before])
                print(json.dumps(dict(two_writers_delivered=True,reader_freed_inside_both_budgets=True,
                    shared_flags_unchanged=True,unicode_exact=True)),flush=True)
            finally:
                for p,fd in zip(processes,fds):send(fd,signal.SIGKILL);p.wait(3);p.stdout.close();os.close(fd)
                for fd in (reader,target,duplicate):os.close(fd)

    def test_valid_stderr_with_closed_stdin_keeps_owned_fds_separate(self):
        reader,target=os.pipe()
        try:
            result=subprocess.run([sys.executable,'-B','-c',
                'import os;from kilix_transcribe.cli import _stderr_write;os.close(0);_stderr_write("日本語\\n",required=True)'],
                stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=target,check=False,timeout=2)
            self.assertEqual((result.returncode,result.stdout),(0,b''))
            self.assertEqual(os.read(reader,100),'日本語\n'.encode())
        finally:os.close(reader);os.close(target)
