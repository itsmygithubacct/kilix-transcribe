"""Word-error rate against a reference, the measure the method was chosen by.

``raw`` keeps lowercase letters, digits and apostrophes. ``normalized`` also
spells out digits, splits hyphens, folds a few spoken forms ("gonna" -> "going
to", "etc" -> "et cetera") and joins compounds that the other side writes as
one word ("data sets"/"datasets"), so spelling style stops counting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
         "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_FOLD = {"gonna": "going to", "wanna": "want to", "gotta": "got to", "ok": "okay",
         "kinda": "kind of", "etc": "et cetera", "'cause": "because"}


def _spell(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + _spell(rest))
    if n < 1_000_000:
        rest = n % 1000
        return _spell(n // 1000) + " thousand" + ("" if rest == 0 else " " + _spell(rest))
    return " ".join(_ONES[int(d)] for d in str(n))


def raw_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))


def normalized_words(text: str) -> list[str]:
    text = text.lower().replace("’", "'").replace("%", " percent")
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"(\d)([a-z])", r"\1 \2", text)
    text = re.sub(r"([a-z])(\d)", r"\1 \2", text)
    text = re.sub(r"\d+", lambda m: " " + _spell(int(m.group())) + " ", text)
    out: list[str] = []
    for word in re.findall(r"[a-z']+", text):
        out.extend(_FOLD.get(word, word).split())
    return out


def _join_compounds(words: list[str], singles: set[str]) -> list[str]:
    out, i = [], 0
    while i < len(words):
        if i + 1 < len(words) and words[i] + words[i + 1] in singles:
            out.append(words[i] + words[i + 1])
            i += 2
        else:
            out.append(words[i])
            i += 1
    return out


@dataclass(frozen=True, slots=True)
class Score:
    reference_words: int
    substitutions: int
    insertions: int
    deletions: int

    @property
    def wer(self) -> float:
        return (self.substitutions + self.insertions + self.deletions) / max(self.reference_words, 1)


def _edits(ref: list[str], hyp: list[str]) -> Score:
    prev = [(j, 0, j, 0) for j in range(len(hyp) + 1)]  # cost, S, I, D
    for i, r in enumerate(ref, 1):
        cur = [(i, 0, 0, i)]
        for j, h in enumerate(hyp, 1):
            c, s, ins, d = prev[j - 1]
            best = (c + (r != h), s + (r != h), ins, d)
            c, s, ins, d = cur[j - 1]
            if c + 1 < best[0]:
                best = (c + 1, s, ins + 1, d)
            c, s, ins, d = prev[j]
            if c + 1 < best[0]:
                best = (c + 1, s, ins, d + 1)
            cur.append(best)
        prev = cur
    _, s, ins, d = prev[-1]
    return Score(len(ref), s, ins, d)


def score(hypothesis: str, reference: str, *, normalized: bool = True) -> Score:
    if not normalized:
        return _edits(raw_words(reference), raw_words(hypothesis))
    ref, hyp = normalized_words(reference), normalized_words(hypothesis)
    ref = _join_compounds(ref, set(hyp))
    hyp = _join_compounds(hyp, set(ref))
    return _edits(ref, hyp)
