"""Private atomic transcript output for provider-owned result names."""

from __future__ import annotations

import os
import re
import stat
import uuid
from pathlib import Path

from .surface import OUTPUTS, SurfaceError, Transcript, render_transcript


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SUFFIX = {"text": ".txt", "json": ".json", "webvtt": ".vtt", "srt": ".srt"}


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise SurfaceError(code, message)


class AtomicTranscriptStore:
    """Commit transcript text beneath one private provider-owned directory."""

    def __init__(self, root: Path):
        _require(isinstance(root, Path) and root.is_absolute(), "OUTPUT_ROOT",
                 "output root must be an absolute Path")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = root.stat(follow_symlinks=False)
        _require(stat.S_ISDIR(info.st_mode), "OUTPUT_ROOT", "output root is not a directory")
        _require(info.st_uid == os.geteuid(), "OUTPUT_ROOT", "output root has another owner")
        _require(info.st_mode & 0o077 == 0, "OUTPUT_ROOT", "output root is not private")
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def commit(self, name: str, transcript: Transcript, output: str) -> Path:
        _require(type(name) is str and _NAME.fullmatch(name) is not None,
                 "OUTPUT_NAME", "output name is outside its safe population")
        _require(output in OUTPUTS, "OUTPUT", "output format is unsupported")
        rendered = render_transcript(transcript, output).encode("utf-8")
        destination = self._root / f"{name}{_SUFFIX[output]}"
        temporary = self._root / f".{name}.{uuid.uuid4().hex}.partial"
        descriptor = -1
        destination_created = False
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            view = memoryview(rendered)
            while view:
                written = os.write(descriptor, view)
                _require(written > 0, "OUTPUT_COMMIT_FAILED", "atomic output write stalled")
                view = view[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            os.link(temporary, destination, follow_symlinks=False)
            destination_created = True
            temporary.unlink()
            directory = os.open(self._root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return destination
        except (OSError, SurfaceError) as error:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            if destination_created:
                destination.unlink(missing_ok=True)
            if isinstance(error, SurfaceError):
                raise
            raise SurfaceError("OUTPUT_COMMIT_FAILED", "atomic output commit failed") from error
