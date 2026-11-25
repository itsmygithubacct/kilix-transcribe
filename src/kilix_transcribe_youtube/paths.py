"""The private data directory: the API key, downloaded audio and run records.

Everything personal stays under one directory that only its owner can read,
``$XDG_DATA_HOME/kilix-transcribe-youtube`` (``~/.local/share/...`` by
default). Nothing here is written into a source checkout.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

APP = "kilix-transcribe-youtube"
KEY_NAME = "minimax_api_key"


class PrivateDataError(RuntimeError):
    """The private directory or the key file is missing or too open."""


def data_root(environ: dict[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    override = env.get("KILIX_TRANSCRIBE_YOUTUBE_HOME")
    if override:
        root = Path(override)
    else:
        base = env.get("XDG_DATA_HOME") or str(Path(env.get("HOME", str(Path.home()))) / ".local" / "share")
        root = Path(base) / APP
    if not root.is_absolute():
        raise PrivateDataError(f"the data directory must be an absolute path: {root}")
    return root


def private_directory(path: Path) -> Path:
    """Create *path* (and parents) and leave it readable only by its owner."""
    if path.is_symlink():
        raise PrivateDataError(f"refusing a symlinked private directory: {path}")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def run_directory(root: Path, video_id: str) -> Path:
    private_directory(root)
    return private_directory(private_directory(root / "runs") / video_id)


def read_key(root: Path) -> str:
    """Read the MiniMax key. The value is never printed or put in an error."""
    path = root / KEY_NAME
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise PrivateDataError(
            f"no MiniMax key at {path}; save it there with mode 600 "
            f"(install -m 600 /path/to/key {path})") from None
    if not stat.S_ISREG(info.st_mode):
        raise PrivateDataError(f"the MiniMax key must be a regular file: {path}")
    if info.st_uid != os.getuid():
        raise PrivateDataError(f"the MiniMax key belongs to another user: {path}")
    if info.st_mode & 0o077:
        raise PrivateDataError(
            f"the MiniMax key is readable by other users; run chmod 600 {path}")
    key = path.read_text(encoding="utf-8").strip()
    if not key or any(ch.isspace() for ch in key):
        raise PrivateDataError(f"the MiniMax key file does not hold one key: {path}")
    return key
