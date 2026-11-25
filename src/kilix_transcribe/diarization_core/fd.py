"""Linux FD-only immutable inputs. No path, network, downloader or launcher."""
import fcntl
import hashlib
import os
import stat

# Linux UAPI; the selected managed CPython 3.12 can omit these names.
F_ADD_SEALS = getattr(fcntl, 'F_ADD_SEALS', 1033)
F_GET_SEALS = getattr(fcntl, 'F_GET_SEALS', 1034)
SEALS = 0x0008 | 0x0004 | 0x0002 | 0x0001

def memfd(label):
    from ..runtime import memory_file
    # memory_file uses the actual libc memfd_create when os omits that API.
    return memory_file()

class Refusal(ValueError):
    pass

def require(test, reason):
    if not test:
        raise Refusal(reason)

def sealed_info(fd, cap, expected_size=None):
    require(type(fd) is int and fd >= 0, 'FD_REQUIRED')
    info = os.fstat(fd)
    require(stat.S_ISREG(info.st_mode), 'REGULAR_SEALED_FD_REQUIRED')
    require(0 <= info.st_size <= cap, 'FD_SIZE_LIMIT')
    try:
        seals = fcntl.fcntl(fd, F_GET_SEALS)
    except OSError as e:
        raise Refusal('IMMUTABLE_SEALS_REQUIRED') from e
    require(seals & SEALS == SEALS, 'IMMUTABLE_SEALS_REQUIRED')
    require(expected_size is None or info.st_size == expected_size, 'INPUT_SIZE_MISMATCH')
    return info

def digest_fd(fd, cap, expected_size=None, expected_sha256=None):
    first = sealed_info(fd, cap, expected_size)
    digest = hashlib.sha256()
    position = 0
    while position < first.st_size:
        part = os.pread(fd, min(65536, first.st_size-position), position)
        require(bool(part), 'TRUNCATED_INPUT')
        digest.update(part)
        position += len(part)
    last = sealed_info(fd, cap, expected_size)
    require((first.st_dev, first.st_ino, first.st_size) ==
            (last.st_dev, last.st_ino, last.st_size), 'INPUT_CHANGED')
    actual = digest.hexdigest()
    require(expected_sha256 is None or actual == expected_sha256, 'INPUT_DIGEST_MISMATCH')
    return actual

def read_sealed(fd, cap):
    info = sealed_info(fd, cap)
    data = os.pread(fd, info.st_size+1, 0)
    require(len(data) == info.st_size, 'TRUNCATED_INPUT')
    return data

def synthetic_fd(data):
    """Inert controls only: visibly named anonymous synthetic FD."""
    fd = memfd('F104-SYNTHETIC-NOT-A-MODEL')
    try:
        position = 0
        while position < len(data):
            position += os.write(fd, data[position:])
        fcntl.fcntl(fd, F_ADD_SEALS, SEALS)
        return fd
    except BaseException:
        os.close(fd)
        raise
