import re
from collections import Counter

from common.rhyme_utils import rhyme_key, split_rhyme_metadata
from syllable.syllabify import count_syllable_range

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_TAG_RE = re.compile(r"<[^>]*>")

ALLOWED_OCTAVE_PATTERNS = {
    "ABBAABBA",  # 42.0%  ABBA+ABBA
    "ABBABAAB",  # 25.7%  ABBA+BAAB
    "ABABABAB",  # 16.0%  ABAB+ABAB
    "ABABBABA",  #  8.1%  ABAB+BABA
    "ABABBAAB",  #  1.8%  ABAB+BAAB
    "ABABABBA",  #  1.4%  ABAB+ABBA
    "ABBAABAB",  #  1.2%  ABBA+ABAB
    "ABBABABA",  #  0.8%  ABBA+BABA
}

ALLOWED_SESTET_PATTERNS = {
    "ABABAB",  # 52.1%  CDC+DCD
    "ABACBC",  # 41.9%  CDC+EDE
    "ABAABA",  #  0.9%  CDC+CDC
    "ABCABC",  #  0.1%  CDE+CDE
    "ABCBAC",  #  n/a   CDE+DCE
    "ABCACB",  #  n/a   CDE+CED
}
ALLOWED_QUATRAIN_PATTERNS = {"ABBA", "ABAB", "BAAB", "BABA"}
ALLOWED_TERCET_PATTERNS = {"ABA", "BAB", "ABC", "BAC", "ACB"}


def _pattern_signature(sequence: str) -> str:
    """Map a rhyme scheme to the canonical form of its pattern"""
    mapping: dict[str, str] = {}
    signature = []

    for symbol in sequence:
        if symbol not in mapping:
            mapping[symbol] = chr(ord("A") + len(mapping) % 26)
        signature.append(mapping[symbol])

    return "".join(signature)


def strip_rhyme_metadata(line: str) -> str:
    """Remove the leading rhyme metadata from a line, if present."""
    tag, suffix, verse = split_rhyme_metadata(line.strip())
    if tag is not None or suffix is not None:
        return verse
    return line


def is_syllable(token: str) -> bool:
    token = token.strip()
    if not token or token.isspace():
        return False

    return any(c.isalpha() for c in token)


def syllable_range(line: str) -> tuple[int, int]:
    """(min, max) syllable count for a verse line:
    - min applies every optional sinalefe
    - max keeps every vowel in hiatus
    """
    clean_line = _TAG_RE.sub(" ", strip_rhyme_metadata(line))
    return count_syllable_range(clean_line)


def count_syllables(line: str) -> int:
    """Lower-bound syllable count for a verse line (every optional sinalefe applied)"""
    return syllable_range(line)[0]


def is_valid_hendecasyllable(line: str, strict: bool = False) -> bool:
    return _range_is_valid(*syllable_range(line), strict)


def _range_is_valid(lo: int, hi: int, strict: bool) -> bool:
    """A line scans as a hendecasyllable when 11 is reachable within its sinalefe range.
    - strict=True -> 11 lies in (lo, hi)
    - strict=False -> (lo, hi) overlaps with [10, 12], i.e. 11 is reachable with at most one optional sinalefe.
    """
    if strict:
        return lo <= 11 <= hi
    return lo <= 12 and hi >= 10


def evaluate_structure(text, strict: bool = False) -> dict:
    text = text.replace("<SONNET>", "").replace("<END>", "").strip()
    text = text.replace("<NEWLINE>", "\n")

    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip() and not line.startswith("<")
    ]

    line_count = len(lines)
    is_14_lines = line_count == 14

    stanzas = [stanza.strip() for stanza in text.split("\n\n") if stanza.strip()]
    stanza_lengths = [
        len([ln for ln in stanza.split("\n") if ln.strip() and not ln.startswith("<")])
        for stanza in stanzas
    ]
    stanza_lengths = [
        stanza_length for stanza_length in stanza_lengths if stanza_length > 0
    ]
    is_correct_structure = stanza_lengths == [4, 4, 3, 3]

    rhyme_map = {}
    current_char = ord("A")
    scheme = []
    syllable_counts = []
    syllable_ranges = []
    valid_hendecasyllables = 0
    for line in lines:
        words = line.split()
        if not words:
            continue
        key = rhyme_key(words[-1])
        if key not in rhyme_map:
            rhyme_map[key] = chr(current_char)
            current_char += 1
        scheme.append(rhyme_map[key])
        lo, hi = syllable_range(line)
        syllable_counts.append(lo)
        syllable_ranges.append((lo, hi))
        if _range_is_valid(lo, hi, strict):
            valid_hendecasyllables += 1

    rhyme_counts = Counter(scheme)
    rhyme_lines = sum(v for v in rhyme_counts.values() if v > 1)

    total_stanzas = len(stanza_lengths)
    valid_stanzas = 0
    cursor = 0

    for i, stanza_length in enumerate(stanza_lengths):
        stanza_scheme = "".join(scheme[cursor : cursor + stanza_length])
        cursor += stanza_length
        stanza_pattern = _pattern_signature(stanza_scheme)
        if i < 2 and stanza_length == 4 and stanza_pattern in ALLOWED_QUATRAIN_PATTERNS:
            valid_stanzas += 1
        elif (
            i >= 2 and stanza_length == 3 and stanza_pattern in ALLOWED_TERCET_PATTERNS
        ):
            valid_stanzas += 1

    is_valid_stanzas = is_14_lines and total_stanzas == 4 and valid_stanzas == 4

    is_valid_rhyme_meter = False
    if is_14_lines and len(scheme) == 14 and valid_hendecasyllables == 14:
        octave_pattern = _pattern_signature("".join(scheme[:8]))
        sestet_pattern = _pattern_signature("".join(scheme[8:]))
        if (
            octave_pattern in ALLOWED_OCTAVE_PATTERNS
            and sestet_pattern in ALLOWED_SESTET_PATTERNS
        ):
            is_valid_rhyme_meter = True

    is_valid_sonnet = is_valid_rhyme_meter and is_correct_structure

    return {
        "is_14_lines": is_14_lines,
        "is_correct_structure": is_correct_structure,
        "is_valid_stanzas": is_valid_stanzas,
        "valid_stanzas": valid_stanzas,
        "total_stanzas": total_stanzas,
        "is_valid_rhyme_meter": is_valid_rhyme_meter,
        "is_valid_sonnet": is_valid_sonnet,
        "rhyme_lines": rhyme_lines,
        "line_count": line_count,
        "valid_hendecasyllables": valid_hendecasyllables,
        "syllable_counts": syllable_counts,
        "syllable_ranges": syllable_ranges,
    }


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion k/n. Default z=1.96 ≈ 95%."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    """Set of lowercased word n-grams in text with struct tags removed."""
    words = _WORD_RE.findall(_TAG_RE.sub(" ", text).lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def ngram_overlap(text: str, reference: set[tuple[str, ...]], n: int) -> float:
    """Fraction of text's word n-grams that also occur in reference.
    1.0 = every n-gram is in the corpus; 0.0 = all novel.
    Returns 0.0 when the text is too short to form a gram.
    """
    grams = word_ngrams(text, n)
    if not grams:
        return 0.0
    return len(grams & reference) / len(grams)
