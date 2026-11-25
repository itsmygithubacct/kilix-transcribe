"""Rolling windows using the pinned CLI's flushed native segment callback."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import select
import subprocess
import sys
import time
import wave
from collections import Counter

from .protocol import ProtocolError, decode_payload, encode_payload
from .streaming import (MAX_WINDOW_BYTES, MAX_UPDATE_TEXT_BYTES, UPDATE_INTERVAL, SegmentStream,
                        worker_frame, CORE_MS, CONTEXT_MS)
from .surface import SegmentUpdate, _segment_payload

RATE = 16000
WINDOW_MS = CORE_MS
PREFIX_MS = 5000
OVERLAP_MS = CONTEXT_MS
_CALLBACK = re.compile(r"^\[(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\]  (.*)$")


def _milliseconds(value):
    hour, minute, seconds = value.split(":")
    second, millisecond = seconds.split(".")
    if int(minute) >= 60 or int(second) >= 60:
        raise ProtocolError("ENGINE_FAILED", "invalid callback timestamp")
    return ((int(hour) * 60 + int(minute)) * 60 + int(second)) * 1000 + int(millisecond)


class Emitter:
    def __init__(self, task):
        self.state = SegmentStream(task)
        self.last = 0.0

    def emit(self, segment_id, start, end, text, stable=False, words=()):
        previous = self.state.segments.get(segment_id)
        revision = 0 if previous is None else previous.revision + 1
        if previous is not None and (previous.text, previous.end_ms, previous.stable) == (text, end, stable):
            return
        update = SegmentUpdate(segment_id, revision, start, end, text, stable, words)
        value = {"sequence": self.state.sequence, "segment": _segment_payload(update)}
        encode_payload(value)  # Enforce the public frame/node limits before writing.
        self.state.apply(value)
        frame = worker_frame({"type": "update", "result": value})
        sys.stdout.buffer.write(frame)
        sys.stdout.buffer.flush()
        self.last = time.monotonic()


def _infer(command, pass_fds, *, emitter, segment_id, base, start, end, limit):
    """Never source an unstable hypothesis from the completed JSON file."""
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, pass_fds=pass_fds)
    parts = []
    consumed = 0
    pending = None
    buffer = bytearray()
    try:
        while True:
            delay = max(0, UPDATE_INTERVAL - (time.monotonic() - emitter.last))
            if pending is not None and delay == 0:
                emitter.emit(segment_id, start, pending[1], pending[0])
                pending = None
            # Flush the single coalesced hypothesis even when inference blocks
            # without producing another line. A blocking readline would defer
            # a real correction until completion.
            if not select.select([process.stdout], [], [], delay if pending is not None else UPDATE_INTERVAL)[0]:
                continue
            block = os.read(process.stdout.fileno(), 4096)
            if not block:
                if buffer:
                    raise ProtocolError("ENGINE_FAILED", "truncated callback line")
                break
            consumed += len(block)
            if consumed > MAX_WINDOW_BYTES:
                raise ProtocolError("LIMIT_EXCEEDED", "engine callback output exceeds bound")
            buffer.extend(block)
            while b"\n" in buffer:
                line, _, remaining = buffer.partition(b"\n")
                buffer = bytearray(remaining)
                text = line.decode("utf-8", "strict")
                if not text.strip():
                    continue
                match = _CALLBACK.fullmatch(text)
                if match is None:
                    raise ProtocolError("ENGINE_FAILED", "unexpected engine callback output")
                t0, t1 = base + _milliseconds(match[1]), base + _milliseconds(match[2])
                if t1 < t0:
                    raise ProtocolError("ENGINE_FAILED", "reversed engine callback time")
                # The pinned engine can round its final boundary past a short
                # audio clip. Clip to actual window duration before ownership.
                t0, t1 = min(t0, limit), min(t1, limit)
                # Callback segments provide provisional overlap ownership;
                # final token timing below assigns it more precisely.
                if not start <= (t0 + t1) // 2 < end:
                    continue
                part = match[3].strip()
                if not part:
                    continue
                parts.append(part)
                text = " ".join(parts)
                if len(text.encode()) > MAX_UPDATE_TEXT_BYTES:
                    raise ProtocolError("LIMIT_EXCEEDED", "callback hypothesis exceeds text bound")
                pending = (text, max(start, min(t1, end)))
                if time.monotonic() - emitter.last >= UPDATE_INTERVAL:
                    emitter.emit(segment_id, start, pending[1], pending[0])
                    pending = None
        if process.wait() != 0:
            raise ProtocolError("ENGINE_FAILED", "streaming inference failed")
        if pending is not None:
            # Coalesce callbacks faster than 20 Hz into one latest revision.
            time.sleep(max(0, UPDATE_INTERVAL - (time.monotonic() - emitter.last)))
            emitter.emit(segment_id, start, pending[1], pending[0])
    finally:
        process.stdout.close()
        # On failure leave the process in the supervisor's owned tree. Its
        # trusted subreaper, not a guessed process group, performs teardown.


def _final_window(payload, *, base, start, end, words, include_end=False, window_end=None,
                  committed=(), assigned=None):
    value = decode_payload(payload, maximum_bytes=MAX_WINDOW_BYTES, maximum_nodes=100_000)
    language = value["result"]["language"]
    units = []
    pending = None
    fallback = []
    remaining = Counter(committed)
    for segment in value["transcription"]:
        for token in segment.get("tokens", []):
            raw = token.get("text", "")
            timing = token.get("offsets")
            if not raw or raw.startswith("[_") or raw.startswith("<|") or not timing:
                continue
            t0, t1 = base + int(timing["from"]), base + int(timing["to"])
            if t1 < t0:
                raise ProtocolError("ENGINE_FAILED", "reversed engine token time")
            if window_end is not None:
                t0, t1 = min(t0, window_end), min(t1, window_end)
            identity = (raw, t0, t1)
            if remaining[identity]:
                remaining[identity] -= 1
                # A visible continuation may already belong to a complete
                # committed unit whose first piece is outside this window.
                # Never glue new pieces across a removed committed unit.
                if pending is not None:
                    units.append(pending)
                    pending = None
                continue
            # Reconstruct complete textual units before discarding overlap.
            # Leading whitespace or the word adapter's >500 ms gap starts
            # another unit; continuation pieces/punctuation stay attached.
            # A run without whitespace is indivisible, not language-specific
            # word segmentation. Retain its exact raw pieces and full timing.
            if pending is not None and (raw[:1].isspace() or t0 > pending[2] + 500):
                units.append(pending)
                pending = None
            token = {"text": raw, "offsets": {"from": t0, "to": t1}}
            if pending is None:
                pending = [[token], t0, t1]
            else:
                pending[0].append(token)
                pending[2] = max(pending[2], t1)
        if not segment.get("tokens"):
            timing = segment["offsets"]
            t0, t1 = base + int(timing["from"]), base + int(timing["to"])
            if start <= (t0 + t1) // 2 < end:
                fallback.append(segment["text"].strip())
    if pending is not None:
        units.append(pending)
    chosen = []
    for tokens, t0, t1 in units:
        midpoint = (t0 + t1) // 2
        if start <= midpoint < end or include_end and midpoint == end:
            chosen.extend(tokens)
    text = ("".join(token["text"] for token in chosen) if chosen else " ".join(fallback)).strip()
    # The stable segment envelope will include boundary-crossing words;
    # clipping here would erase part of their original full-window timing.
    word_start = min([start, *(token["offsets"]["from"] for token in chosen)])
    word_end = max([end, *(token["offsets"]["to"] for token in chosen)])
    if assigned is not None:
        assigned.extend((token["text"], token["offsets"]["from"], token["offsets"]["to"])
                        for token in chosen)
    return language, text, words(chosen, word_start, word_end)


def rolling_segments(decoded: Path, directory: Path, frames, command, pass_fds, task, words):
    duration = (frames * 1000 + RATE - 1) // RATE
    emitter = Emitter(task)
    language = "auto"
    segments = []
    committed = []
    with wave.open(str(decoded), "rb") as reader:
        for segment_id, start in enumerate(range(0, duration, WINDOW_MS)):
            end = min(duration, start + WINDOW_MS)
            base = max(0, start - OVERLAP_MS)
            committed = [identity for identity in committed if identity[2] >= base]
            full_end = min(duration, end + OVERLAP_MS)
            prefix_end = min(full_end, start + PREFIX_MS)
            passes = [prefix_end, full_end] if prefix_end < full_end else [full_end]
            for index, stop in enumerate(passes):
                window = directory / "window.wav"
                destination = directory / "window-result"
                reader.setpos(base * RATE // 1000)
                pcm = reader.readframes(min(frames - reader.tell(), (stop - base) * RATE // 1000))
                with wave.open(str(window), "wb") as writer:
                    writer.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
                    writer.writeframes(pcm)
                invocation = list(command)
                invocation[invocation.index("--file") + 1] = str(window)
                invocation[invocation.index("--output-file") + 1] = str(destination)
                _infer(invocation, pass_fds, emitter=emitter, segment_id=segment_id,
                       base=base, start=start, end=min(end, stop), limit=stop)
                result_file = destination.with_suffix(".json")
                if index == len(passes) - 1:
                    with result_file.open("rb") as source:
                        payload = source.read(MAX_WINDOW_BYTES + 1)
                    assigned = []
                    language, text, timings = _final_window(payload, base=base, start=start, end=end,
                        words=words, include_end=end == duration, window_end=stop,
                        committed=committed, assigned=assigned)
                    stable_start = min([start, *(word.start_ms for word in timings)])
                    stable_end = max([end, *(word.end_ms for word in timings)])
                    emitter.emit(segment_id, stable_start, stable_end, text, True, timings)
                    committed.extend(assigned)
                    segments.append(emitter.state.segments[segment_id])
                result_file.unlink()
                window.unlink()
    return language, segments
