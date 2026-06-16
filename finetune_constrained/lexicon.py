import os
import re
from dataclasses import dataclass, field

from common.rhyme_utils import rhyme_key, split_rhyme_metadata

_STRIP = '.,;:!?"()[]«»“”‘…—–- \t'
_HAS_ALPHA = re.compile(r"[^\W\d_]", re.UNICODE)


@dataclass
class Word:
    text: str
    key: str
    freq: int


@dataclass
class Lexicon:
    by_key: dict[str, list[Word]]
    _tok_cache: dict[str, list[int]] = field(default_factory=dict)

    def class_size(self, key: str) -> int:
        return len(self.by_key.get(key, ()))

    def rich_keys(self, min_size: int) -> list[str]:
        """Rhyme keys with at least ``min_size`` distinct words."""
        return [k for k, ws in self.by_key.items() if len(ws) >= min_size]

    def candidates_pool(self, min_size: int, exclude: set[str], cap: int) -> list[Word]:
        """Candidate words for a line that defines a new rhyme:
        - drawn only from rich classes,
        - whose key isn't already used in this sonnet,
        - ranked by corpus frequency and capped."""
        pool: list[Word] = []
        for k in self.rich_keys(min_size):
            if k in exclude:
                continue
            pool.extend(self.by_key[k])
        pool.sort(key=lambda w: w.freq, reverse=True)
        return pool[:cap]


def _iter_line_final_words(data_dir: str):
    for fn in sorted(os.listdir(data_dir)):
        if not fn.endswith(".txt"):
            continue
        with open(os.path.join(data_dir, fn), encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                _, _, verse = split_rhyme_metadata(line)
                verse = verse.strip()
                if not verse or verse.startswith("<"):
                    continue
                tokens = verse.split()
                if not tokens:
                    continue
                word = tokens[-1].strip(_STRIP)
                if len(word) < 2 or not _HAS_ALPHA.search(word):
                    continue
                yield word


def build_lexicon(data_dir: str) -> Lexicon:
    counts: dict[str, dict[str, int]] = {}
    for word in _iter_line_final_words(data_dir):
        key = rhyme_key(word)
        if not key:
            continue
        counts.setdefault(key, {}).setdefault(word, 0)
        counts[key][word] += 1

    by_key: dict[str, list[Word]] = {}
    for key, words in counts.items():
        ws = [Word(text=w, key=key, freq=c) for w, c in words.items()]
        ws.sort(key=lambda w: w.freq, reverse=True)
        by_key[key] = ws
    return Lexicon(by_key=by_key)
