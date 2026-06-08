from syllable.syllabify import (
    _merge_kind,
    count_syllable_range,
    syllabify_text,
    syllabify_word,
)


def test_syllabify_word_basic():
    assert syllabify_word("casa") == ["ca", "sa"]
    assert syllabify_word("sposa") == ["spo", "sa"]


def test_count_syllable_range_single_words_have_zero_width():
    assert count_syllable_range("casa") == (2, 2)
    assert count_syllable_range("luna mare") == (4, 4)


def test_count_syllable_range_optional_sinalefe_widens():
    assert count_syllable_range("io e") == (1, 2)


def test_silent_h_triggers_sinalefe():
    assert _merge_kind("co ", "ha ") == "sinalefe"
    assert _merge_kind("co ", "sa ") is None


def test_apostrophe_is_obligatory_elision():
    assert _merge_kind("co' ", "altri") == "elision"


def test_undercounted_piano_line_now_scans_strict():
    line = "per ingeggnacce inzieme io e la sposa"
    lo, hi = count_syllable_range(line)
    assert lo <= 11 <= hi


def test_syllabify_text_is_lossless_on_letters():
    text = "<SONNET>\nMe fo sposo, Taddeo. Quer zantarello\n<END>"
    recon = "".join(syllabify_text(text))
    assert [c for c in recon if c.isalpha()] == [c for c in text if c.isalpha()]
