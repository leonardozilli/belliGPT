import re
import unicodedata

ACCENTED_VOWELS = "àèéìíòóùú"
PLAIN_VOWELS = "aeiou"

_RHYME_TAG = r"(?:<RHYME_[A-Z]>|[Ⓐ-Ⓩ])"
RHYME_METADATA_RE = re.compile(
    rf"^\s*(?P<tag>{_RHYME_TAG})?\s*"
    rf"(?:(?P<suffix>[^\s|]+)\s*\|\s*)?"
    rf"(?P<verse>.*)$"
)


def split_rhyme_metadata(line: str) -> tuple[str | None, str | None, str]:
    """Split a single line into (tag, suffix, verse)"""
    match = RHYME_METADATA_RE.match(line)
    assert match is not None
    return match.group("tag"), match.group("suffix"), match.group("verse")


def normalize_word(word: str) -> str:
    """Remove accents, punctuation, and lowercase the word."""
    word = unicodedata.normalize("NFD", word)
    word = re.sub(r"[^\w\s]", "", word)
    word = "".join(c for c in word if unicodedata.category(c) != "Mn")
    return word.lower()


def extract_rhyme_suffix(word: str) -> str:
    # normalize curly apostrophes to straight ones
    word = word.replace("’", "'").lower()

    # if compound word, focus on the last chunk
    parts = re.split(r"['-]", word)
    target_chunk = parts[-1]

    if not target_chunk:
        return ""

    # srip non-alpha chars
    original = re.sub(rf"[^\w\s{ACCENTED_VOWELS}]", "", target_chunk)
    normalized = normalize_word(target_chunk)

    if not normalized:
        return ""

    # search for an accented vowel and return from there to end
    for i in range(len(original) - 1, -1, -1):
        if original[i] in ACCENTED_VOWELS:
            return original[i:]

    # if no accented vowel, use plain vowel logic
    vowel_indices = [i for i, c in enumerate(normalized) if c in PLAIN_VOWELS]

    # if no vowels, take the last 3 chars
    if not vowel_indices:
        return normalized[-3:]

    # if vowels, take the last 2 plus any following chars
    start = vowel_indices[-2] if len(vowel_indices) >= 2 else vowel_indices[0]

    return normalized[start:]


def rhyme_key(word: str) -> str:
    return normalize_word(extract_rhyme_suffix(word))
