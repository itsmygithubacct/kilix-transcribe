"""MiniMax asr-1.0 over audio cut into pieces under the service's 500 s cap."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ENDPOINT = "https://api.minimax.io/v1/speech_to_text"
MODEL = "asr-1.0"
PIECE_SECONDS = 420
SAMPLE_RATE = "16000"
MAX_PIECE_SECONDS = 490
MAX_PIECE_BYTES = 45_000_000
ATTEMPTS = 5
# Plan, balance and authorisation refusals: retrying cannot help.
LIMIT_CODES = {1004, 1008, 2056}
LIMIT_WORDS = ("usage limit", "token plan", "quota", "insufficient", "balance", "payment")


class MiniMaxError(RuntimeError):
    pass


class MiniMaxLimit(MiniMaxError):
    """The account refused the request; earlier pieces stay saved."""


@dataclass(frozen=True, slots=True)
class Segment:
    start_ms: int
    end_ms: int
    text: str


# (url, body, headers, timeout) -> (http status, response text)
Transport = Callable[[str, bytes, dict[str, str], float], tuple[int, str]]


def urllib_transport(url: str, body: bytes, headers: dict[str, str], timeout: float) -> tuple[int, str]:
    request = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", errors="replace")


def probe_seconds(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def cut_pieces(audio: Path, directory: Path) -> list[Path]:
    """Mono 16 kHz 64 kb/s MP3 pieces of at most PIECE_SECONDS; reused when present.

    16 kHz matters: the same video cut at YouTube's 48 kHz scored 1.00% word
    errors against its uploader captions, and 0.66% at 16 kHz."""
    directory.mkdir(mode=0o700, exist_ok=True)
    existing = sorted(directory.glob("piece-*.mp3"))
    if existing:
        return existing
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(audio),
         "-ac", "1", "-ar", SAMPLE_RATE, "-b:a", "64k", "-f", "segment", "-segment_time", str(PIECE_SECONDS),
         "-reset_timestamps", "1", str(directory / "piece-%03d.mp3")],
        check=True)
    pieces = sorted(directory.glob("piece-*.mp3"))
    if not pieces:
        raise MiniMaxError(f"ffmpeg cut no pieces from {audio.name}")
    for piece in pieces:
        if probe_seconds(piece) > MAX_PIECE_SECONDS or piece.stat().st_size > MAX_PIECE_BYTES:
            raise MiniMaxError(f"{piece.name} is over the service's size or length cap")
    return pieces


def _multipart(fields: dict[str, str], filename: str, payload: bytes) -> tuple[str, bytes]:
    boundary = "----kilixtranscribeyoutube"
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        for name, value in fields.items()
    ]
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
        "Content-Type: audio/mpeg\r\n\r\n".encode())
    parts.append(payload)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return boundary, b"".join(parts)


def _is_limit(code: int, message: str) -> bool:
    return code in LIMIT_CODES or any(word in message.lower() for word in LIMIT_WORDS)


def transcribe_piece(key: str, piece: Path, *, response_format: str = "json",
                     transport: Transport = urllib_transport,
                     sleep: Callable[[float], None] = time.sleep, language: str = "en") -> dict:
    """One piece. Network and server errors are retried; the host's resolver
    fails intermittently, so a name-resolution error is too.

    ``json`` returns the words; ``verbose_json`` adds segment times but its
    words differ, consistently and for the worse (it expands contractions the
    reference keeps), so it is asked for only when no other timing exists."""
    if response_format not in ("json", "verbose_json"):
        raise ValueError(f"unknown response format {response_format!r}")
    boundary, body = _multipart(
        {"model": MODEL, "response_format": response_format, "stream": "false"},
        piece.name, piece.read_bytes())
    headers = {"Authorization": f"Bearer {key}", "language": language,
               "Content-Type": f"multipart/form-data; boundary={boundary}"}
    delay = 10.0
    last = "no attempt made"
    for attempt in range(1, ATTEMPTS + 1):
        try:
            status, raw = transport(ENDPOINT, body, headers, 600.0)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last = f"network error: {getattr(error, 'reason', error)}"
        else:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {}
            base = payload.get("base_resp") if isinstance(payload, dict) else None
            code = int(base.get("status_code") or 0) if isinstance(base, dict) else 0
            message = str(base.get("status_msg") or "") if isinstance(base, dict) else ""
            if status == 200 and code == 0 and isinstance(payload.get("text"), str):
                return payload
            detail = message or raw[:200]
            if status in (401, 402, 403) or _is_limit(code, detail):
                raise MiniMaxLimit(f"MiniMax refused the request (HTTP {status}, status {code}): {detail}")
            if status in (400, 413):
                raise MiniMaxError(f"MiniMax rejected {piece.name} (HTTP {status}, status {code}): {detail}")
            last = f"HTTP {status}, status {code}: {detail}"
        if attempt < ATTEMPTS:
            sleep(delay)
            delay = min(delay * 2, 120.0)
    raise MiniMaxError(f"MiniMax did not transcribe {piece.name} after {ATTEMPTS} attempts ({last})")


def _cached(key: str, piece: Path, response_format: str, **kwargs) -> dict:
    kind = "text" if response_format == "json" else "segments"
    saved = piece.with_name(f"{piece.stem}.{kind}.json")
    if saved.is_file():
        return json.loads(saved.read_text(encoding="utf-8"))
    payload = transcribe_piece(key, piece, response_format=response_format, **kwargs)
    saved.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    saved.chmod(0o600)
    return payload


def _onto_segments(words: list[str], timed: list[dict], duration: float) -> list[tuple[float, float, str]]:
    """Share *words* out over *timed* segments in proportion to their word
    counts: the two texts differ only in a few words, so a word's position
    in one is its position in the other to within a word or two."""
    spans = [(float(s.get("start", 0.0)), float(s.get("end", 0.0)), len(str(s.get("text", "")).split()))
             for s in timed]
    spans = [span for span in spans if span[2] > 0]
    if not spans or not words:
        return [(0.0, duration, " ".join(words))] if words else []
    total = sum(count for _, _, count in spans)
    out, taken = [], 0
    for index, (start, end, count) in enumerate(spans):
        share = len(words) - taken if index == len(spans) - 1 else round(len(words) * count / total)
        piece_words = words[taken:taken + share]
        taken += len(piece_words)
        if piece_words:
            out.append((start, end, " ".join(piece_words)))
    return out


def transcribe(key: str, audio: Path, work: Path, *, segment_times: bool = False,
               transport: Transport = urllib_transport,
               sleep: Callable[[float], None] = time.sleep,
               cutter: Callable[[Path, Path], list[Path]] | None = None,
               seconds: Callable[[Path], float] = probe_seconds,
               progress: Callable[[str], None] = lambda line: None) -> list[Segment]:
    """Segments on the whole audio's clock, one per piece unless
    *segment_times* asks MiniMax for its own segment times as well. Each reply
    is saved, so a run stopped by a limit or an error resumes where it stopped."""
    # The directory names the cutting settings, so pieces cut another way are
    # never reused for these.
    pieces = (cutter or cut_pieces)(audio, work / f"pieces-{SAMPLE_RATE}")
    segments: list[Segment] = []
    offset_ms = 0
    options = {"transport": transport, "sleep": sleep}
    for index, piece in enumerate(pieces, 1):
        payload = _cached(key, piece, "json", **options)
        duration = float(payload.get("duration") or seconds(piece))
        words = str(payload.get("text", "")).split()
        if segment_times:
            timed = _cached(key, piece, "verbose_json", **options).get("segments") or []
            spans = _onto_segments(words, timed, duration)
        else:
            spans = [(0.0, duration, " ".join(words))] if words else []
        progress(f"minimax piece {index}/{len(pieces)}")
        for start_s, end_s, text in spans:
            start = offset_ms + round(start_s * 1000)
            end = offset_ms + round(end_s * 1000)
            segments.append(Segment(start, max(start, end), text))
        offset_ms += round(duration * 1000)
    return segments
