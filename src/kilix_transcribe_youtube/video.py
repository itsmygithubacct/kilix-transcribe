"""Recognise a YouTube video id in an id or a URL."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

_ID = re.compile(r"[A-Za-z0-9_-]{11}")
_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
          "youtu.be", "www.youtube-nocookie.com", "youtube-nocookie.com"}


class VideoError(ValueError):
    pass


def video_id(value: str) -> str:
    value = value.strip()
    if _ID.fullmatch(value):
        return value
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or parsed.hostname not in _HOSTS:
        raise VideoError(f"not a YouTube video id or URL: {value!r}")
    if parsed.hostname == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif parsed.path == "/watch":
        candidate = (parse_qs(parsed.query).get("v") or [""])[0]
    else:
        parts = [part for part in parsed.path.split("/") if part]
        candidate = parts[1] if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v") else ""
    if not _ID.fullmatch(candidate):
        raise VideoError(f"no video id in {value!r}")
    return candidate


def watch_url(vid: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id(vid)}"
