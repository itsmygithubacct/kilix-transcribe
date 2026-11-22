"""Download a video's audio and English caption tracks with yt-dlp.

Captions and audio are separate requests, with a pause between subtitle
downloads: asking for several tracks in one burst gets HTTP 429 from YouTube.
Only the original-language automatic track (``en-orig``) and an uploader track
are requested, never machine translations.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from kilix_transcribe_youtube.video import watch_url

UPLOADER_LANGS = ("en-US", "en-GB", "en")
AUTO_LANG = "en-orig"
ATTEMPTS = 3

Runner = Callable[[list[str]], subprocess.CompletedProcess]


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Fetched:
    audio: Path
    auto_captions: Path | None
    uploader_captions: Path | None


@dataclass(frozen=True, slots=True)
class SearchHit:
    video_id: str
    title: str
    channel: str
    duration: str


def yt_dlp_path(environ: dict[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    if env.get("KILIX_YT_DLP"):
        return env["KILIX_YT_DLP"]
    local = Path(env.get("HOME", str(Path.home()))) / ".local" / "bin" / "yt-dlp"
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    found = shutil.which("yt-dlp")
    if not found:
        raise FetchError("yt-dlp is not installed; put a current release at ~/.local/bin/yt-dlp")
    return found


def _run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True)


# What yt-dlp writes for "-f bestaudio". Anything else in the run directory
# (captions, an info.json, a .part file) is never mistaken for the audio.
AUDIO_SUFFIXES = (".webm", ".m4a", ".opus", ".ogg", ".mp3", ".mka", ".aac", ".mp4")


def _downloaded_audio(directory: Path) -> Path | None:
    return next((p for p in sorted(directory.glob("audio.*"))
                 if p.suffix in AUDIO_SUFFIXES and p.is_file()), None)


class Fetcher:
    def __init__(self, executable: str | None = None, *, runner: Runner = _run,
                 sleep: Callable[[float], None] = time.sleep,
                 cookies_from_browser: str | None = None) -> None:
        self.executable = executable or yt_dlp_path()
        self.runner = runner
        self.sleep = sleep
        self.cookies = cookies_from_browser

    def _base(self) -> list[str]:
        command = [self.executable, "--no-warnings", "--no-progress", "--no-playlist"]
        if self.cookies:
            command += ["--cookies-from-browser", self.cookies]
        return command

    def _call(self, arguments: list[str], what: str) -> subprocess.CompletedProcess:
        delay = 20.0
        result = None
        for attempt in range(1, ATTEMPTS + 1):
            result = self.runner(self._base() + arguments)
            if result.returncode == 0:
                return result
            if "429" not in (result.stderr or "") or attempt == ATTEMPTS:
                break
            self.sleep(delay)
            delay *= 2
        tail = (result.stderr or "").strip().splitlines()[-1:] if result else []
        raise FetchError(f"yt-dlp could not {what}: {tail[0] if tail else 'no detail'}")

    def search(self, query: str, count: int = 5) -> list[SearchHit]:
        result = self._call(
            ["--flat-playlist", "--print", "%(id)s\t%(title)s\t%(channel)s\t%(duration_string)s",
             f"ytsearch{count}:{query}"], "search YouTube")
        hits = []
        for line in result.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) == 4:
                hits.append(SearchHit(*parts))
        return hits

    def fetch(self, video_id: str, directory: Path) -> Fetched:
        url = watch_url(video_id)
        template = str(directory / "audio.%(ext)s")
        marker = directory / ".captions-fetched"
        if not marker.is_file():
            # Two requests: with --write-auto-subs, "en" would select YouTube's
            # machine translation, so uploader tracks and the original-language
            # automatic track are asked for separately.
            self._call(["--skip-download", "--write-subs", "--sub-langs", ",".join(UPLOADER_LANGS),
                        "--sub-format", "json3", "--sleep-subtitles", "5", "-o", template, url],
                       "download the uploader captions")
            self._call(["--skip-download", "--write-auto-subs", "--sub-langs", AUTO_LANG,
                        "--sub-format", "json3", "--sleep-subtitles", "5", "-o", template, url],
                       "download the automatic captions")
            marker.touch()
        audio = _downloaded_audio(directory)
        if audio is None:
            self._call(["-f", "bestaudio", "-o", template, url], "download the audio")
            audio = _downloaded_audio(directory)
        if audio is None:
            raise FetchError(f"yt-dlp reported success but wrote no audio for {video_id}")
        for path in directory.iterdir():
            if path.is_file() and not path.is_symlink():
                path.chmod(0o600)
        auto = directory / f"audio.{AUTO_LANG}.json3"
        uploader = next((directory / f"audio.{lang}.json3" for lang in UPLOADER_LANGS
                         if (directory / f"audio.{lang}.json3").is_file()), None)
        return Fetched(audio, auto if auto.is_file() else None, uploader)
