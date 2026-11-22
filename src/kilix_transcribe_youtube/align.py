"""Time each MiniMax word from YouTube's word-timed automatic captions.

MiniMax returns its words in segments about a minute long, which is too coarse
for subtitles. YouTube's automatic track times every word but gets more words
wrong. Aligning the two keeps MiniMax's words and borrows YouTube's timing:
words the tracks agree on take the caption timing, and the words between two
agreements are spread evenly across the gap. Without a caption track, words are
spread across their MiniMax segment by length.
"""

from __future__ import annotations

import difflib
import re
from typing import Sequence

from kilix_transcribe_youtube.captions import TimedWord
from kilix_transcribe_youtube.minimax import Segment

# A caption anchor more than this far outside its MiniMax segment is a chance
# match elsewhere in the video, not the same word.
ANCHOR_SLACK_MS = 15_000


def _key(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())


def _spread(count: int, start: int, end: int) -> list[tuple[int, int]]:
    if count == 0:
        return []
    end = max(start, end)
    step = (end - start) / count
    return [(round(start + i * step), round(start + (i + 1) * step)) for i in range(count)]


def _by_length(segment: Segment) -> list[TimedWord]:
    words = segment.text.split()
    total = sum(len(word) + 1 for word in words) or 1
    span = segment.end_ms - segment.start_ms
    out, clock = [], float(segment.start_ms)
    for word in words:
        width = span * (len(word) + 1) / total
        out.append(TimedWord(round(clock), round(clock + width), word))
        clock += width
    return out


def align(segments: Sequence[Segment], captions: Sequence[TimedWord] = ()) -> list[TimedWord]:
    flat: list[tuple[str, Segment]] = [(word, seg) for seg in segments for word in seg.text.split()]
    if not flat:
        return []
    if not captions:
        return [word for seg in segments for word in _by_length(seg)]

    keys = [_key(word) for word, _ in flat]
    cap_keys = [_key(word.text) for word in captions]
    anchors: dict[int, TimedWord] = {}
    matcher = difflib.SequenceMatcher(None, keys, cap_keys, autojunk=False)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            i, j = block.a + offset, block.b + offset
            if not keys[i]:
                continue
            seg, cap = flat[i][1], captions[j]
            if seg.start_ms - ANCHOR_SLACK_MS <= cap.start_ms <= seg.end_ms + ANCHOR_SLACK_MS:
                anchors[i] = cap

    timed: list[TimedWord | None] = [None] * len(flat)
    for i, cap in anchors.items():
        timed[i] = TimedWord(cap.start_ms, cap.end_ms, flat[i][0])

    # Fill each run of unanchored words between its neighbouring anchors (or
    # the ends of its own segments when the run reaches the start or end).
    i = 0
    while i < len(flat):
        if timed[i] is not None:
            i += 1
            continue
        j = i
        while j < len(flat) and timed[j] is None:
            j += 1
        left = timed[i - 1].end_ms if i > 0 else flat[i][1].start_ms
        right = timed[j].start_ms if j < len(flat) else flat[j - 1][1].end_ms
        for k, (start, end) in zip(range(i, j), _spread(j - i, left, right)):
            timed[k] = TimedWord(start, end, flat[k][0])
        i = j

    # Captions can overlap slightly; keep the sequence non-decreasing.
    result: list[TimedWord] = []
    floor = 0
    for word in timed:
        assert word is not None
        start = max(word.start_ms, floor)
        end = max(word.end_ms, start)
        result.append(TimedWord(start, end, word.text))
        floor = start
    return result
