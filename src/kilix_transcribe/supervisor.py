"""Dedicated Linux descendant owner; never reaps the provider's other children."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

stopping = False


def stop(_signal, _frame):
    global stopping
    stopping = True


def reap_descendants():
    """A dedicated subreaper owns adoptees even after setsid/setpgid."""
    children_file = Path(f"/proc/self/task/{os.getpid()}/children")
    while True:
        # Only this supervisor's direct children can appear here. Killing
        # parents reparents any still-running descendants to this subreaper.
        children = [int(value) for value in children_file.read_text().split()]
        for child in children:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
        while True:
            try:
                reaped, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if reaped == 0:
                break
        time.sleep(0.005)


def main():
    parent = os.getppid()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    libc = ctypes.CDLL(None, use_errno=True)
    # PR_SET_CHILD_SUBREAPER, PR_SET_PDEATHSIG, PR_SET_NO_NEW_PRIVS.
    if (libc.prctl(36, 1, 0, 0, 0) != 0
            or libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0
            or libc.prctl(38, 1, 0, 0, 0) != 0):
        return 125
    if os.getppid() != parent or parent == 1:
        return 125
    payload = sys.stdin.buffer.read(65_537)
    if len(payload) > 65_536:
        return 125
    request = json.loads(payload)
    descriptors = list(request.get("runtime_fds", {}).values())
    if request.get("audio_fd") is not None:
        descriptors.append(request["audio_fd"])
    worker = subprocess.Popen(
        [sys.executable, "-I", "-B", str(Path(__file__).with_name("worker.py"))],
        stdin=subprocess.PIPE, stdout=sys.stdout.buffer, stderr=subprocess.DEVNULL,
        pass_fds=tuple(descriptors), start_new_session=True,
    )
    try:
        assert worker.stdin is not None
        worker.stdin.write(payload)
        worker.stdin.close()
        while worker.poll() is None and not stopping:
            time.sleep(0.01)
        return 130 if stopping else worker.returncode
    finally:
        reap_descendants()


if __name__ == "__main__":
    raise SystemExit(main())
