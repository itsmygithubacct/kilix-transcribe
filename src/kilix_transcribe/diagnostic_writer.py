"""Bounded local diagnostic writes; no child process or borrowed-flag mutation."""
from __future__ import annotations

import os
import select
import stat
import time


def deliver(descriptor: int, data: bytes, expires: float) -> bool:
    """Open an independent nonblocking writer for this exact Linux FD target.

    A dup shares status flags. Reopening /proc/self/fd gives pipes and terminals
    their own open-file description, so a full receiver cannot block this job
    or change another writer. Unsupported targets refuse without a helper.
    """
    if not data:
        return True
    if time.monotonic() >= expires:
        return False
    owned = None
    try:
        before = os.fstat(descriptor)
        if not (stat.S_ISFIFO(before.st_mode) or stat.S_ISCHR(before.st_mode)
                or stat.S_ISREG(before.st_mode)):
            return False
        owned = os.open(f"/proc/self/fd/{descriptor}",
                        os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOCTTY)
        after = os.fstat(owned)
        if ((before.st_dev, before.st_ino, before.st_mode, before.st_rdev)
                != (after.st_dev, after.st_ino, after.st_mode, after.st_rdev)):
            return False
        return write_deadline(owned, data, expires)
    except OSError:
        return False
    finally:
        if owned is not None:
            os.close(owned)


def write_deadline(descriptor: int, data: bytes, expires: float) -> bool:
    """Retry transient receiver pressure without changing descriptor flags."""
    view = memoryview(data)
    while view:
        if time.monotonic() >= expires:
            return False
        # POSIX pipes guarantee at least 512 atomic bytes. Keep each complete
        # UTF-8 codepoint within a chunk; successful partial writes retain the
        # exact remaining byte population rather than restarting a message.
        end = min(512, len(view))
        while end < len(view) and view[end] & 0xC0 == 0x80:
            end -= 1
        try:
            count = os.write(descriptor, view[:end])
        except InterruptedError:
            continue
        except BlockingIOError:
            remaining = expires - time.monotonic()
            if remaining <= 0:
                return False
            try:
                if not select.select([], [descriptor], [], remaining)[1]:
                    return False
            except InterruptedError:
                continue
            except (OSError, ValueError):
                return False
            # Another writer can consume readiness before our next attempt.
            continue
        except OSError:
            return False
        if count <= 0 or count > end:
            return False
        view = view[count:]
    return True
