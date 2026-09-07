"""Strict result documents bound to control metadata and source requests."""
from __future__ import annotations

from .protocol import ProtocolError, decode_payload
from .surface import RESULT_SCHEMA, SegmentUpdate, SurfaceError, Transcript, WordTiming, render_transcript

MAX_RESULT_BYTES = 16 * 1024 * 1024


def decode_result(payload: bytes, metadata: dict, arguments: dict) -> dict:
    try:
        value = decode_payload(payload, maximum_bytes=MAX_RESULT_BYTES, maximum_nodes=1_000_000)
        expected = {"schema", "engine", "model", "language", "segments", "task",
                    "engine_revision", "model_revision", "duration_ms", "text", "output_format", "output"}
        if set(value) != expected or value["schema"] != RESULT_SCHEMA:
            raise ValueError("invalid result population")
        if (type(value["engine"]) is not dict or set(value["engine"]) != {"id"}
                or type(value["model"]) is not dict or set(value["model"]) != {"id"}
                or type(value["engine_revision"]) is not str or not 0 < len(value["engine_revision"]) <= 128
                or type(value["model_revision"]) is not str or not 0 < len(value["model_revision"]) <= 128
                or type(value["duration_ms"]) is not int or not 0 < value["duration_ms"] <= 3_600_000
                or type(value["segments"]) is not list):
            raise ValueError("invalid result identity or duration")
        for name in ("engine", "model"):
            if value[name]["id"] != metadata.get(f"{name}_id"):
                raise ValueError("result identity mismatch")
        for name in ("engine_revision", "model_revision", "duration_ms"):
            if type(metadata.get(name)) is not type(value[name]) or value[name] != metadata[name]:
                raise ValueError("result metadata mismatch")
        if value["task"] != arguments["task"] or value["output_format"] != arguments["output"]:
            raise ValueError("result request mismatch")
        segments = []
        for segment in value["segments"]:
            required = {"segment_id", "revision", "start_ms", "end_ms", "text", "stable", "words"}
            if type(segment) is not dict or not required <= set(segment) <= required | {"speaker"}:
                raise ValueError("invalid segment fields")
            if type(segment["words"]) is not list or segment["end_ms"] > value["duration_ms"]:
                raise ValueError("invalid segment population or time")
            words = []
            for word in segment["words"]:
                if (type(word) is not dict or not {"start_ms", "end_ms", "text"} <= set(word)
                        <= {"start_ms", "end_ms", "text", "confidence"}):
                    raise ValueError("invalid word fields")
                words.append(WordTiming(word["start_ms"], word["end_ms"], word["text"], word.get("confidence")))
            speaker = segment.get("speaker", {})
            if type(speaker) is not dict or (speaker and set(speaker) != {"id", "confidence"}):
                raise ValueError("invalid speaker fields")
            segments.append(SegmentUpdate(segment["segment_id"], segment["revision"], segment["start_ms"],
                                          segment["end_ms"], segment["text"], segment["stable"], tuple(words),
                                          speaker.get("id"), speaker.get("confidence")))
        transcript = Transcript(value["task"], value["engine"]["id"], value["model"]["id"],
                                value["language"], tuple(segments))
        if (value["text"] != render_transcript(transcript, "text")
                or value["output"] != render_transcript(transcript, value["output_format"])):
            raise ValueError("serialized result mismatch")
        return value
    except ProtocolError as error:
        code = "LIMIT_EXCEEDED" if error.code == "LIMIT_EXCEEDED" else "INVALID_RESPONSE"
        raise ProtocolError(code, "malformed transcript result") from error
    except (SurfaceError, ValueError, TypeError, KeyError) as error:
        raise ProtocolError("INVALID_RESPONSE", "malformed transcript result") from error
