"""Group timed words into subtitle-sized cues on the kilix-transcribe surface."""

from __future__ import annotations

from typing import Sequence

from kilix_transcribe.surface import SegmentUpdate, Transcript, WordTiming

from kilix_transcribe_youtube import ENGINE_ID, MODEL_ID
from kilix_transcribe_youtube.captions import TimedWord

MAX_CUE_CHARS = 84      # two 42-character subtitle lines
MAX_CUE_MS = 7_000
GAP_BREAK_MS = 1_500    # a pause this long starts a new cue
SENTENCE_END = (".", "?", "!")


def cues(words: Sequence[TimedWord]) -> list[list[TimedWord]]:
    groups: list[list[TimedWord]] = []
    current: list[TimedWord] = []
    for word in words:
        if current:
            text_len = sum(len(w.text) + 1 for w in current) + len(word.text)
            too_long = text_len > MAX_CUE_CHARS or word.end_ms - current[0].start_ms > MAX_CUE_MS
            paused = word.start_ms - current[-1].end_ms >= GAP_BREAK_MS
            sentence_done = current[-1].text.endswith(SENTENCE_END) and text_len > MAX_CUE_CHARS // 2
            if too_long or paused or sentence_done:
                groups.append(current)
                current = []
        current.append(word)
    if current:
        groups.append(current)
    return groups


def transcript(words: Sequence[TimedWord], *, language: str = "en") -> Transcript:
    segments = []
    for index, group in enumerate(cues(words)):
        start, end = group[0].start_ms, max(w.end_ms for w in group)
        segments.append(SegmentUpdate(
            segment_id=index, revision=0, start_ms=start, end_ms=end,
            text=" ".join(w.text for w in group), stable=True,
            words=tuple(WordTiming(w.start_ms, min(w.end_ms, end), w.text) for w in group)))
    return Transcript(task="transcribe", engine_id=ENGINE_ID, model_id=MODEL_ID,
                      language=language, segments=tuple(segments))
