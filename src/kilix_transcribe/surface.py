"""Pure, engine-neutral transcription mechanics.

The types in this module are an implementation candidate.  They deliberately
do not open a socket, microphone, model, source tree, or accelerator.  No type
here is the P1 wire/catalog contract, and synthetic identities used by tests do
not select a P2 engine or release profile.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


SURFACE_SCHEMA = "kilix.transcribe.surface/candidate-v1"
RESULT_SCHEMA = "kilix.transcribe.result/candidate-v1"
COMMANDS = ("record", "file", "serve", "models", "status", "cancel", "unload")
OUTPUTS = ("text", "json", "webvtt", "srt")
TASKS = ("transcribe", "translate", "diarize")
RUNTIME_COMMANDS = ("record", "file", "serve", "cancel", "unload")
PROVIDER_REFUSAL = (
    "KILIX_TRANSCRIBE_REFUSAL [RUNTIME_UNSELECTED] "
    "no transcription runtime or release profile is selected"
)

# Candidate-only safety bounds.  They are local mechanics, not F100/F106
# fields, release thresholds, or an accepted P1 contract.
MAX_SEGMENTS = 100_000
MAX_WORDS_PER_SEGMENT = 100_000
MAX_RESULT_WORDS = 100_000
MAX_SEGMENT_TEXT_BYTES = 1_048_576
MAX_RESULT_TEXT_BYTES = 16_777_216
MAX_ID_BYTES = 256

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class SurfaceError(ValueError):
    """A stable candidate refusal with a machine-readable reason code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class RuntimeUnselected(SurfaceError):
    """Raised when an operational command reaches the unselected boundary."""

    def __init__(self) -> None:
        super().__init__("RUNTIME_UNSELECTED", PROVIDER_REFUSAL)


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise SurfaceError(code, message)


def _exact_nonnegative_int(value: object, code: str, name: str) -> int:
    _require(type(value) is int and value >= 0, code, f"{name} must be a non-negative integer")
    return value


def _finite_confidence(value: object, code: str, name: str) -> float:
    _require(type(value) in {float, int} and type(value) is not bool,
             code, f"{name} must be numeric")
    numeric = float(value)
    _require(math.isfinite(numeric) and 0.0 <= numeric <= 1.0,
             code, f"{name} must be finite and within 0..1")
    return numeric


def _bounded_identity(value: object, code: str, name: str) -> str:
    _require(isinstance(value, str) and value.strip() == value and bool(value),
             code, f"{name} must be a non-empty trimmed string")
    _require(not _CONTROL_CHARACTERS.search(value), code,
             f"{name} contains a forbidden control character")
    _require(len(value.encode("utf-8")) <= MAX_ID_BYTES, code,
             f"{name} exceeds the candidate identity bound")
    return value


def _bounded_text(value: object, code: str, name: str) -> str:
    _require(isinstance(value, str), code, f"{name} must be text")
    _require(not _CONTROL_CHARACTERS.search(value), code,
             f"{name} contains a forbidden control character")
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    _require(
        not normalized.startswith("\n")
        and not normalized.endswith("\n")
        and "\n\n" not in normalized,
        code,
        f"{name} contains a blank subtitle line",
    )
    _require(len(normalized.encode("utf-8")) <= MAX_SEGMENT_TEXT_BYTES,
             code, f"{name} exceeds the candidate text bound")
    return normalized


@dataclass(frozen=True, slots=True)
class WordTiming:
    """A bounded word timing emitted by an eventual engine adapter."""

    start_ms: int
    end_ms: int
    text: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        start_ms = _exact_nonnegative_int(self.start_ms, "WORD_TIME", "word start_ms")
        end_ms = _exact_nonnegative_int(self.end_ms, "WORD_TIME", "word end_ms")
        _require(end_ms >= start_ms, "WORD_TIME", "word end_ms precedes start_ms")
        object.__setattr__(self, "text", _bounded_text(self.text, "WORD_TEXT", "word text"))
        if self.confidence is not None:
            object.__setattr__(
                self,
                "confidence",
                _finite_confidence(self.confidence, "WORD_CONFIDENCE", "word confidence"),
            )


@dataclass(frozen=True, slots=True)
class SegmentUpdate:
    """One revision of a transcript segment.

    Unstable revisions may be replaced.  A stable revision is immutable and is
    the only kind admitted to a final Transcript.
    """

    segment_id: int
    revision: int
    start_ms: int
    end_ms: int
    text: str
    stable: bool
    words: tuple[WordTiming, ...] = ()
    speaker: str | None = None
    speaker_confidence: float | None = None

    def __post_init__(self) -> None:
        _exact_nonnegative_int(self.segment_id, "SEGMENT_ID", "segment_id")
        _exact_nonnegative_int(self.revision, "SEGMENT_REVISION", "revision")
        start_ms = _exact_nonnegative_int(self.start_ms, "SEGMENT_TIME", "segment start_ms")
        end_ms = _exact_nonnegative_int(self.end_ms, "SEGMENT_TIME", "segment end_ms")
        _require(end_ms >= start_ms, "SEGMENT_TIME", "segment end_ms precedes start_ms")
        _require(type(self.stable) is bool, "SEGMENT_STABILITY", "stable must be boolean")
        object.__setattr__(self, "text", _bounded_text(self.text, "SEGMENT_TEXT", "segment text"))

        _require(isinstance(self.words, tuple), "WORD_POPULATION", "words must be a tuple")
        _require(len(self.words) <= MAX_WORDS_PER_SEGMENT, "WORD_POPULATION",
                 "word population exceeds the candidate bound")
        previous_start = start_ms
        for word in self.words:
            _require(isinstance(word, WordTiming), "WORD_TYPE",
                     "every word must be a WordTiming")
            _require(start_ms <= word.start_ms <= word.end_ms <= end_ms,
                     "WORD_TIME", "word timing is outside its segment")
            _require(word.start_ms >= previous_start, "WORD_ORDER",
                     "word timings are not ordered by start_ms")
            previous_start = word.start_ms

        paired_speaker_fields = (self.speaker is None) == (self.speaker_confidence is None)
        _require(paired_speaker_fields, "SPEAKER_PAIR",
                 "speaker and speaker_confidence must appear together")
        if self.speaker is not None:
            object.__setattr__(
                self,
                "speaker",
                _bounded_identity(self.speaker, "SPEAKER_ID", "speaker"),
            )
            object.__setattr__(
                self,
                "speaker_confidence",
                _finite_confidence(
                    self.speaker_confidence,
                    "SPEAKER_CONFIDENCE",
                    "speaker_confidence",
                ),
            )


@dataclass(frozen=True, slots=True)
class Transcript:
    """A final, stable result suitable for deterministic serialization."""

    task: str
    engine_id: str
    model_id: str
    language: str
    segments: tuple[SegmentUpdate, ...]

    def __post_init__(self) -> None:
        _require(type(self.task) is str and self.task in TASKS,
                 "TASK", "task is outside the 3/3 candidate population")
        object.__setattr__(
            self, "engine_id", _bounded_identity(self.engine_id, "ENGINE_ID", "engine_id")
        )
        object.__setattr__(
            self, "model_id", _bounded_identity(self.model_id, "MODEL_ID", "model_id")
        )
        object.__setattr__(
            self, "language", _bounded_identity(self.language, "LANGUAGE", "language")
        )
        _require(isinstance(self.segments, tuple), "SEGMENT_POPULATION",
                 "segments must be a tuple")
        _require(len(self.segments) <= MAX_SEGMENTS, "SEGMENT_POPULATION",
                 "segment population exceeds the candidate bound")

        identities: set[int] = set()
        total_text_bytes = 0
        total_words = 0
        for segment in self.segments:
            _require(isinstance(segment, SegmentUpdate), "SEGMENT_TYPE",
                     "every segment must be a SegmentUpdate")
            _require(segment.stable, "UNSTABLE_RESULT",
                     "final results may contain only stable segments")
            _require(segment.segment_id not in identities, "SEGMENT_DUPLICATE",
                     "final result contains a duplicate segment_id")
            identities.add(segment.segment_id)
            total_text_bytes += len(segment.text.encode("utf-8"))
            total_words += len(segment.words)
            _require(total_words <= MAX_RESULT_WORDS, "WORD_POPULATION",
                     "aggregate word population exceeds the candidate bound")
            total_text_bytes += sum(len(word.text.encode("utf-8")) for word in segment.words)
            _require(total_text_bytes <= MAX_RESULT_TEXT_BYTES, "RESULT_TEXT",
                     "result text exceeds the candidate bound")


class TranscriptAssembler:
    """Apply unstable replacements and seal only a completely stable result."""

    def __init__(self, task: str) -> None:
        _require(type(task) is str and task in TASKS,
                 "TASK", "task is outside the 3/3 candidate population")
        self._task = task
        self._segments: dict[int, SegmentUpdate] = {}
        self._sealed = False

    @property
    def pending_segments(self) -> int:
        return sum(not segment.stable for segment in self._segments.values())

    def apply(self, update: SegmentUpdate) -> None:
        _require(not self._sealed, "ASSEMBLER_SEALED", "the assembler is already sealed")
        _require(isinstance(update, SegmentUpdate), "SEGMENT_TYPE",
                 "update must be a SegmentUpdate")
        current = self._segments.get(update.segment_id)
        if current is None:
            _require(update.revision == 0, "SEGMENT_REVISION",
                     "the first revision for a segment must be zero")
            _require(len(self._segments) < MAX_SEGMENTS, "SEGMENT_POPULATION",
                     "segment population exceeds the candidate bound")
        else:
            _require(not current.stable, "SEGMENT_STABLE",
                     "a stable segment cannot be replaced")
            _require(update.revision == current.revision + 1, "SEGMENT_REVISION",
                     "replacement revisions must advance by exactly one")
        self._segments[update.segment_id] = update

    def finish(self, *, engine_id: str, model_id: str, language: str) -> Transcript:
        _require(not self._sealed, "ASSEMBLER_SEALED", "the assembler is already sealed")
        _require(self.pending_segments == 0, "UNSTABLE_RESULT",
                 "cannot finish while an unstable segment remains")
        ordered = tuple(
            sorted(
                self._segments.values(),
                key=lambda segment: (segment.start_ms, segment.end_ms, segment.segment_id),
            )
        )
        result = Transcript(
            task=self._task,
            engine_id=engine_id,
            model_id=model_id,
            language=language,
            segments=ordered,
        )
        self._sealed = True
        return result


def _word_payload(word: WordTiming) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "end_ms": word.end_ms,
        "start_ms": word.start_ms,
        "text": word.text,
    }
    if word.confidence is not None:
        payload["confidence"] = word.confidence
    return payload


def _segment_payload(segment: SegmentUpdate) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "end_ms": segment.end_ms,
        "revision": segment.revision,
        "segment_id": segment.segment_id,
        "stable": True,
        "start_ms": segment.start_ms,
        "text": segment.text,
        "words": [_word_payload(word) for word in segment.words],
    }
    if segment.speaker is not None:
        payload["speaker"] = {
            "confidence": segment.speaker_confidence,
            "id": segment.speaker,
        }
    return payload


def _format_timestamp(milliseconds: int, separator: str) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


def _escape_cue(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _bounded_rendered(value: str) -> str:
    _require(len(value.encode("utf-8")) <= MAX_RESULT_TEXT_BYTES,
             "RESULT_TEXT", "serialized result exceeds the candidate byte bound")
    return value


def render_transcript(transcript: Transcript, output: str) -> str:
    """Render a final result to one of the exact 4/4 candidate formats."""

    _require(isinstance(transcript, Transcript), "RESULT_TYPE",
             "transcript must be a final Transcript")
    _require(type(output) is str and output in OUTPUTS,
             "OUTPUT", "output is outside the 4/4 candidate population")

    if output == "text":
        text = "\n".join(segment.text for segment in transcript.segments)
        return _bounded_rendered(f"{text}\n" if text else "")

    if output == "json":
        payload = {
            "engine": {"id": transcript.engine_id},
            "language": transcript.language,
            "model": {"id": transcript.model_id},
            "schema": RESULT_SCHEMA,
            "segments": [_segment_payload(segment) for segment in transcript.segments],
            "task": transcript.task,
        }
        rendered = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ) + "\n"
        return _bounded_rendered(rendered)

    separator = "." if output == "webvtt" else ","
    blocks: list[str] = []
    for index, segment in enumerate(transcript.segments, start=1):
        start = _format_timestamp(segment.start_ms, separator)
        end = _format_timestamp(segment.end_ms, separator)
        blocks.append(f"{index}\n{start} --> {end}\n{_escape_cue(segment.text)}")

    if output == "webvtt":
        return _bounded_rendered(
            "WEBVTT\n\n" + "\n\n".join(blocks) + ("\n" if blocks else "")
        )
    return _bounded_rendered("\n\n".join(blocks) + ("\n" if blocks else ""))


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel-requested"
    SUCCEEDED = "succeeded"
    CANCELED = "canceled"
    FAILED = "failed"


class JobLifecycle:
    """A deterministic job state machine with one terminal result at most."""

    __slots__ = ("_failure_code", "_result", "_state")

    def __init__(self) -> None:
        self._state = JobState.QUEUED
        self._result: Transcript | None = None
        self._failure_code: str | None = None

    @property
    def state(self) -> JobState:
        return self._state

    @property
    def result(self) -> Transcript | None:
        return self._result

    @property
    def failure_code(self) -> str | None:
        return self._failure_code

    def start(self) -> JobState:
        _require(self.state is JobState.QUEUED, "JOB_TRANSITION",
                 "only a queued job can start")
        self._state = JobState.RUNNING
        return self.state

    def request_cancel(self) -> JobState:
        if self.state is JobState.QUEUED:
            self._state = JobState.CANCELED
        elif self.state is JobState.RUNNING:
            self._state = JobState.CANCEL_REQUESTED
        elif self.state not in {JobState.CANCEL_REQUESTED, JobState.CANCELED}:
            raise SurfaceError("JOB_TERMINAL", "a terminal job cannot be canceled")
        return self.state

    def acknowledge_cancel(self) -> JobState:
        _require(self.state is JobState.CANCEL_REQUESTED, "JOB_TRANSITION",
                 "only a cancel-requested job can acknowledge cancellation")
        self._state = JobState.CANCELED
        return self.state

    def complete(self, result: Transcript) -> JobState:
        _require(self.state is JobState.RUNNING, "JOB_TRANSITION",
                 "only a running job can complete")
        _require(isinstance(result, Transcript), "RESULT_TYPE",
                 "completion requires a final Transcript")
        self._result = result
        self._state = JobState.SUCCEEDED
        return self.state

    def fail(self, code: str) -> JobState:
        _require(self.state in {JobState.QUEUED, JobState.RUNNING, JobState.CANCEL_REQUESTED},
                 "JOB_TERMINAL", "a terminal job cannot fail again")
        self._failure_code = _bounded_identity(code, "FAILURE_CODE", "failure code")
        self._result = None
        self._state = JobState.FAILED
        return self.state


def status_payload() -> dict[str, Any]:
    return {
        "engine_routes": {"selected": 0, "total": 4},
        "provider_state": "RUNTIME_UNSELECTED",
        "release_profiles": {"selected": 0, "total": 1},
        "schema": SURFACE_SCHEMA,
        "source_objects": {"selected": 0, "total": 8},
    }


def models_payload() -> dict[str, Any]:
    return {
        "engine_routes": {"selected": 0, "total": 4},
        "models": [],
        "schema": SURFACE_SCHEMA,
        "source_objects": {"selected": 0, "total": 8},
    }


def inspect_command(command: str) -> dict[str, Any]:
    """Serve static introspection and fail closed for all runtime operations."""

    _require(command in COMMANDS, "COMMAND", "command is outside the 7/7 population")
    if command == "status":
        return status_payload()
    if command == "models":
        return models_payload()
    raise RuntimeUnselected()


def synthetic_transcript(updates: Iterable[SegmentUpdate]) -> Transcript:
    """Build the checker fixture without giving synthetic identities authority."""

    assembler = TranscriptAssembler("transcribe")
    for update in updates:
        assembler.apply(update)
    return assembler.finish(
        engine_id="synthetic-engine-not-selected",
        model_id="synthetic-model-not-selected",
        language="en",
    )
