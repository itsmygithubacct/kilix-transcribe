"""Negotiated, bounded segment revisions and private worker framing."""
from __future__ import annotations

import json
import struct

from .protocol import MAX_CONTROL_FRAME_BYTES, STREAM_FEATURE, ProtocolError, decode_payload
from .surface import (SegmentUpdate, SurfaceError, TranscriptAssembler, WordTiming,
                      _segment_payload)

MAX_UPDATES = 4096
MAX_STREAM_SEGMENTS = 180
MAX_UPDATE_TEXT_BYTES = 32768
MAX_UPDATE_WORDS = 256
MAX_STREAM_TEXT_BYTES = 1024 * 1024
MAX_WINDOW_BYTES = 1024 * 1024
UPDATE_INTERVAL = 0.05
SEND_TIMEOUT = 0.2
CORE_MS = 20000
CONTEXT_MS = 2000
MAX_DURATION_MS = 3_600_000
STREAM_LIMITS = {"frame_bytes": MAX_CONTROL_FRAME_BYTES, "updates": MAX_UPDATES,
                 "segments": MAX_STREAM_SEGMENTS, "text_bytes": MAX_UPDATE_TEXT_BYTES,
                 "words": MAX_UPDATE_WORDS, "update_interval_ms": 50,
                 "send_timeout_ms": 200}
_U32 = struct.Struct("!I")


def segment_from_payload(value: dict) -> SegmentUpdate:
    try:
        fields = {"segment_id", "revision", "start_ms", "end_ms", "text", "stable", "words"}
        if type(value) is not dict or set(value) != fields or type(value["words"]) is not list:
            raise ValueError("invalid segment population")
        if (type(value["text"]) is not str
                or len(value["text"].encode("utf-8")) > MAX_UPDATE_TEXT_BYTES
                or len(value["words"]) > MAX_UPDATE_WORDS):
            raise ProtocolError("LIMIT_EXCEEDED", "update exceeds text or word bound")
        words = []
        for word in value["words"]:
            if (type(word) is not dict or not {"start_ms", "end_ms", "text"} <= set(word)
                    <= {"start_ms", "end_ms", "text", "confidence"}):
                raise ValueError("invalid word fields")
            words.append(WordTiming(word["start_ms"], word["end_ms"], word["text"], word.get("confidence")))
        update = SegmentUpdate(value["segment_id"], value["revision"], value["start_ms"],
                               value["end_ms"], value["text"], value["stable"], tuple(words))
        if (update.segment_id >= MAX_STREAM_SEGMENTS or update.revision >= MAX_UPDATES
                or update.end_ms > MAX_DURATION_MS):
            raise ProtocolError("LIMIT_EXCEEDED", "update identity or time exceeds bound")
        return update
    except (SurfaceError, ValueError, TypeError, KeyError, UnicodeError) as error:
        if isinstance(error, ProtocolError):
            raise
        raise ProtocolError("INVALID_RESPONSE", "invalid transcript update") from error


class SegmentStream:
    """Validate the whole revision history, then bind it to the final result."""
    def __init__(self, task):
        self.assembler = TranscriptAssembler(task)
        self.segments = {}
        self.sequence = 0
        self.text_bytes = 0
        self.word_count = 0

    def apply(self, value):
        if (type(value) is not dict or set(value) != {"sequence", "segment"}
                or type(value["sequence"]) is not int or value["sequence"] != self.sequence):
            raise ProtocolError("INVALID_RESPONSE", "non-contiguous update sequence")
        if self.sequence >= MAX_UPDATES:
            raise ProtocolError("LIMIT_EXCEEDED", "update count exceeds bound")
        update = segment_from_payload(value["segment"])
        current = self.segments.get(update.segment_id)
        if current is None:
            if update.segment_id != len(self.segments):
                raise ProtocolError("INVALID_RESPONSE", "non-contiguous rolling core identity")
            if self.segments:
                previous = self.segments[update.segment_id - 1]
                if not previous.stable or previous.end_ms < update.segment_id * CORE_MS:
                    raise ProtocolError("INVALID_RESPONSE", "rolling core advanced before legal stabilization")
                # A following core proves the previous nominal end was not
                # EOF. Recheck before accepting or exposing this new update.
                self._validate_core(previous, allow_eof=False)
        elif update.segment_id != len(self.segments) - 1:
            raise ProtocolError("INVALID_RESPONSE", "revision returned to an earlier rolling core")
        self._validate_core(update)
        size = len(update.text.encode()) + sum(len(word.text.encode()) for word in update.words)
        previous_size = (len(current.text.encode()) + sum(len(word.text.encode()) for word in current.words)
                         if current is not None else 0)
        total = self.text_bytes + size - previous_size
        count = self.word_count + len(update.words) - (len(current.words) if current is not None else 0)
        if total > MAX_STREAM_TEXT_BYTES or count > MAX_STREAM_SEGMENTS * MAX_UPDATE_WORDS:
            raise ProtocolError("LIMIT_EXCEEDED", "aggregate transcript exceeds bound")
        try:
            self.assembler.apply(update)
        except SurfaceError as error:
            raise ProtocolError("INVALID_RESPONSE", "invalid revision or stability transition") from error
        self.segments[update.segment_id] = update
        self.sequence += 1
        self.text_bytes, self.word_count = total, count
        return update

    @staticmethod
    def _validate_core(update, *, allow_eof=True):
        start = update.segment_id * CORE_MS
        end = start + CORE_MS
        word_start = min([start, *(word.start_ms for word in update.words)])
        word_end = max([end, *(word.end_ms for word in update.words)])
        if update.stable:
            legal = (update.start_ms == word_start and update.start_ms >= max(0, start - CONTEXT_MS)
                     and start <= update.end_ms <= min(word_end, end + CONTEXT_MS))
        else:
            legal = update.start_ms == start and start <= update.end_ms <= end
        if not legal:
            raise ProtocolError("INVALID_RESPONSE", "update time does not belong to its rolling core")
        for word in update.words:
            midpoint = (word.start_ms + word.end_ms) // 2
            if not (start <= midpoint < end or allow_eof and word.start_ms == word.end_ms == end):
                raise ProtocolError("INVALID_RESPONSE", "word belongs to a different rolling core")

    def finish(self, result):
        try:
            duration = result["duration_ms"]
            if type(duration) is not int or not 0 < duration <= MAX_DURATION_MS:
                raise ValueError("invalid final duration")
            if self.segments and len(self.segments) != (duration + CORE_MS - 1) // CORE_MS:
                raise ValueError("final duration disagrees with rolling core population")
            for identity, segment in self.segments.items():
                end = min(duration, (identity + 1) * CORE_MS)
                if segment.end_ms != max([end, *(word.end_ms for word in segment.words)]):
                    raise ValueError("stable core does not cover its final ownership interval")
                for word in segment.words:
                    midpoint = (word.start_ms + word.end_ms) // 2
                    if word.end_ms > duration or not (midpoint < end or word.start_ms == word.end_ms == end == duration):
                        raise ValueError("word exceeds final ownership interval")
            transcript = self.assembler.finish(engine_id=result["engine"]["id"],
                model_id=result["model"]["id"], language=result["language"])
            if [_segment_payload(segment) for segment in transcript.segments] != result["segments"]:
                raise ValueError("final result disagrees with stable updates")
        except (SurfaceError, ValueError, KeyError, TypeError) as error:
            raise ProtocolError("INVALID_RESPONSE", "uncommitted or inconsistent stream result") from error


def worker_frame(value, *, maximum_bytes=MAX_CONTROL_FRAME_BYTES):
    # The final private pipe frame may be larger than a control packet. Public
    # finals still travel in the existing descriptor-based result envelope.
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    if not 0 < len(payload) <= maximum_bytes:
        raise ProtocolError("LIMIT_EXCEEDED", "worker frame exceeds bound")
    return _U32.pack(len(payload)) + payload


class WorkerFrames:
    """An incremental pipe decoder retaining at most one bounded frame."""
    def __init__(self, maximum_bytes):
        self.maximum_bytes = maximum_bytes
        self.buffer = bytearray()
        self.length = None

    @property
    def wanted(self):
        return (4 if self.length is None else self.length) - len(self.buffer)

    def feed(self, block):
        if len(block) > self.wanted:
            raise ProtocolError("ENGINE_FAILED", "worker framing overflow")
        self.buffer.extend(block)
        if self.wanted:
            return None
        if self.length is None:
            self.length = _U32.unpack(self.buffer)[0]
            self.buffer.clear()
            if not 0 < self.length <= self.maximum_bytes:
                raise ProtocolError("LIMIT_EXCEEDED", "worker frame exceeds bound")
            return None
        payload = bytes(self.buffer)
        self.buffer.clear()
        self.length = None
        return decode_payload(payload, maximum_bytes=self.maximum_bytes, maximum_nodes=1_000_000)

    def eof(self):
        if self.buffer or self.length is not None:
            raise ProtocolError("ENGINE_FAILED", "truncated worker frame")
