"""Read YouTube json3 caption tracks into timed words."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TimedWord:
    start_ms: int
    end_ms: int
    text: str


def timed_words(path: Path) -> list[TimedWord]:
    """Words of a json3 track, each with a start and an end in milliseconds.

    A word starts at its event's ``tStartMs`` plus its ``tOffsetMs`` and ends
    where the next word starts, or at its event's end. Events that only append
    a line break carry no words.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    starts: list[tuple[int, int, str]] = []  # start, event end, text
    for event in data.get("events", []):
        segs = event.get("segs")
        if not segs:
            continue
        base = int(event.get("tStartMs", 0))
        event_end = base + int(event.get("dDurationMs", 0))
        for seg in segs:
            for piece in seg.get("utf8", "").split():
                starts.append((base + int(seg.get("tOffsetMs", 0)), event_end, piece))
    starts.sort(key=lambda item: item[0])
    words: list[TimedWord] = []
    for index, (start, event_end, text) in enumerate(starts):
        following = starts[index + 1][0] if index + 1 < len(starts) else event_end
        end = max(start, min(following, event_end))
        words.append(TimedWord(start, end, text))
    return words


def plain_text(path: Path) -> str:
    return " ".join(word.text for word in timed_words(path))
