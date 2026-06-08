from common.eval import (
    count_syllables,
    evaluate_structure,
    is_syllable,
    is_valid_hendecasyllable,
    ngram_overlap,
    strip_rhyme_metadata,
    syllable_range,
    wilson_interval,
    word_ngrams,
)


def test_count_syllables_matches_validity():
    line = "ma nun vojo piú affríggeme nun vojjo"
    assert count_syllables(line) == 11
    assert is_valid_hendecasyllable(line, strict=True) is True
    assert count_syllables(f"à | {line}") == 11


def test_evaluate_structure_exposes_syllable_counts():
    metrics = evaluate_structure("casa\nmare\nluna")
    assert metrics["syllable_counts"] == [2, 2, 2]
    # one count per scored line
    assert len(metrics["syllable_counts"]) == metrics["line_count"]
    assert metrics["syllable_ranges"] == [(2, 2), (2, 2), (2, 2)]


def test_syllable_range_strips_metadata_and_tags():
    line = "ma nun vojo piú affríggeme nun vojjo"
    lo, hi = syllable_range(line)
    assert lo <= 11 <= hi
    # rhyme metadata / special tokens must not inflate the range
    assert syllable_range(f"à | {line}") == (lo, hi)


def test_range_validity_recovers_undercounted_line():
    # a line whose greedy single-count was 10 but whose range contains 11 must pass even in strict mode.
    line = "per ingeggnacce inzieme io e la sposa"
    lo, hi = syllable_range(line)
    assert lo < 11 <= hi  # 11 reachable only via the upper bound
    assert is_valid_hendecasyllable(line, strict=True) is True


def test_range_validity_rejects_unreachable_count():
    # a short line whose whole range sits below 10 is invalid either way.
    assert is_valid_hendecasyllable("casa mare", strict=False) is False
    assert is_valid_hendecasyllable("casa mare", strict=True) is False


def test_wilson_interval():
    assert wilson_interval(0, 0) == (0.0, 0.0)
    low, high = wilson_interval(15, 30)
    assert low < 0.5 < high
    lo0, hi0 = wilson_interval(0, 30)
    assert lo0 == 0.0 and 0.0 < hi0 < 1.0
    lo1, hi1 = wilson_interval(30, 30)
    assert hi1 == 1.0 and 0.0 < lo1 < 1.0


def test_ngram_overlap_and_word_ngrams():
    ref = word_ngrams("the quick brown fox jumps over", 3)
    assert ngram_overlap("the quick brown fox", ref, 3) == 1.0
    assert ngram_overlap("a totally novel phrase this one is", ref, 3) == 0.0
    # tags are stripped before n-gramming
    assert word_ngrams("<RHYME_A> casa mare", 2) == {("casa", "mare")}
    # too short to form an n-gram
    assert ngram_overlap("casa", ref, 3) == 0.0


def _build_sonnet_from_scheme(scheme: str) -> str:
    words = {
        "A": "casa",
        "B": "mare",
        "C": "luna",
        "D": "sole",
        "E": "vento",
    }

    lines = [
        f"Linea {index + 1} termina con {words[letter]}"
        for index, letter in enumerate(scheme)
    ]

    return "\n\n".join(
        [
            "\n".join(lines[0:4]),
            "\n".join(lines[4:8]),
            "\n".join(lines[8:11]),
            "\n".join(lines[11:14]),
        ]
    )


def test_evaluate_structure_valid_sonnet():
    text = """<SONNET>
    Te lo saressi creso, eh Gurgumella,
    ch’er zor paìno, er zor dorce-me-frega,
    che mmanco ha ffiato per annà a bbottega,
    potessi slargà er buscio a ’na zitella?

    Tu nu lo sai ch’edè sta marachella;
    tutta farina de quell’antra strega.
    Mo che nun trova lei chi jje la sega,
    fa la ruffiana de la su’ sorella.

    Io sarebbe omo, corpo de l’abbrei,
    senza mettécce né ssale né ojjo,
    de dàjjene tre vorte trentasei:

    ma nun vojo piú affríggeme nun vojjo;
    che de donne pe ddio come che llei
    ’ggni monnezzaro me ne dà un pricojjo.
    """

    metrics = evaluate_structure(text)

    assert metrics["is_14_lines"] is True
    assert metrics["is_correct_structure"] is True
    assert metrics["valid_stanzas"] == 4
    assert metrics["total_stanzas"] == 4
    assert metrics["line_count"] == 14
    assert metrics["rhyme_lines"] == 14
    assert metrics["is_valid_sonnet"] is True


def test_evaluate_structure_valid_sonnet_not_strict():
    text = """<SONNET>
    Jeri, all’orloggio de la Cchiesa Nova,
    fra Luca incontrò Agnesa co la brocca.
    Dice: «Beato lui», dice, «a chi tocca»,
    dice, «e nun sa ch’edè chi nu lo prova».

    Risponne lei, dice: «Chi cerca, trova;
    ma a me», dice, «puliteve la bocca».
    «Aùh», dicéee... «e perché nun te fai biocca?»
    «Eh», dice, «e chi me mette sotto l’ova?»

    «Ce n’ho io», dice, «un paro fresche vive»,
    dice, «e ttamante, e tutt’e ddua ’ngallate:
    le vôi sperà si ssò bbone o ccattive?»

    Checco, te pensi che nun l’ha pijjate?
    Ah llei pe nnun sapé legge né scrive,
    ha vorzuto assaggià l’ova der frate.
    """

    metrics = evaluate_structure(text, strict=False)

    assert metrics["is_14_lines"] is True
    assert metrics["is_correct_structure"] is True
    assert metrics["valid_stanzas"] == 4
    assert metrics["total_stanzas"] == 4
    assert metrics["line_count"] == 14
    assert metrics["rhyme_lines"] == 14
    assert metrics["is_valid_sonnet"] is True


def test_evaluate_structure_invalid_sonnet():
    text = """
    Ciarivò ir Papa, ch’er Papa io me sce vorze
    la fasse scusa der Papa Palazzo,
    si cce s’abbi er monno doppo er gran birba-de-staggione a ppoco la pupazzone
    """

    metrics = evaluate_structure(text)

    assert metrics["is_14_lines"] is False
    assert metrics["is_correct_structure"] is False
    assert metrics["valid_stanzas"] == 0
    assert metrics["total_stanzas"] == 1
    assert metrics["is_valid_sonnet"] is False
    assert metrics["line_count"] == 3


def test_evaluate_structure_rejects_valid_octave_with_invalid_sestet():
    text = _build_sonnet_from_scheme("ABBAABBAEEEEEE")

    metrics = evaluate_structure(text)

    assert metrics["is_14_lines"] is True
    assert metrics["is_correct_structure"] is True
    assert metrics["valid_stanzas"] == 2
    assert metrics["total_stanzas"] == 4
    assert metrics["is_valid_sonnet"] is False


def test_rhyme_grouping_collapses_accent_variants():
    text = "perché\nmare\nmare\ncaffè"
    metrics = evaluate_structure(text)
    assert metrics["rhyme_lines"] == 4


def test_is_valid_sonnet_requires_stanza_structure():
    sonnet = """Te lo saressi creso, eh Gurgumella,
    ch’er zor paìno, er zor dorce-me-frega,
    che mmanco ha ffiato per annà a bbottega,
    potessi slargà er buscio a ’na zitella?
    Tu nu lo sai ch’edè sta marachella;
    tutta farina de quell’antra strega.
    Mo che nun trova lei chi jje la sega,
    fa la ruffiana de la su’ sorella.
    Io sarebbe omo, corpo de l’abbrei,
    senza mettécce né ssale né ojjo,
    de dàjjene tre vorte trentasei:
    ma nun vojo piú affríggeme nun vojjo;
    che de donne pe ddio come che llei
    ’ggni monnezzaro me ne dà un pricojjo."""

    metrics = evaluate_structure(sonnet)

    assert metrics["is_14_lines"] is True
    assert metrics["is_correct_structure"] is False
    assert metrics["is_valid_rhyme_meter"] is True
    assert metrics["is_valid_sonnet"] is False


def test_is_valid_sonnet_with_correct_stanzas():
    sonnet = "\n\n".join(
        [
            "Te lo saressi creso, eh Gurgumella,\n"
            "ch’er zor paìno, er zor dorce-me-frega,\n"
            "che mmanco ha ffiato per annà a bbottega,\n"
            "potessi slargà er buscio a ’na zitella?",
            "Tu nu lo sai ch’edè sta marachella;\n"
            "tutta farina de quell’antra strega.\n"
            "Mo che nun trova lei chi jje la sega,\n"
            "fa la ruffiana de la su’ sorella.",
            "Io sarebbe omo, corpo de l’abbrei,\n"
            "senza mettécce né ssale né ojjo,\n"
            "de dàjjene tre vorte trentasei:",
            "ma nun vojo piú affríggeme nun vojjo;\n"
            "che de donne pe ddio come che llei\n"
            "’ggni monnezzaro me ne dà un pricojjo.",
        ]
    )

    metrics = evaluate_structure(sonnet)

    assert metrics["is_correct_structure"] is True
    assert metrics["is_valid_rhyme_meter"] is True
    assert metrics["is_valid_sonnet"] is True


def test_is_syllable_accepts_letters_rejects_symbols():
    assert is_syllable("ella") is True
    assert is_syllable("à") is True
    assert is_syllable("×") is False
    assert is_syllable("÷") is False
    assert is_syllable(",") is False
    assert is_syllable("   ") is False


def test_strip_rhyme_metadata_plain_suffix():
    assert (
        strip_rhyme_metadata("ella | Te lo saressi creso, eh Gurgumella,")
        == "Te lo saressi creso, eh Gurgumella,"
    )


def test_strip_rhyme_metadata_accented_suffix():
    assert (
        strip_rhyme_metadata("à | che mmanco ha ffiato per annà a bbottega,")
        == "che mmanco ha ffiato per annà a bbottega,"
    )
    assert (
        strip_rhyme_metadata("òcca | fra Luca incontrò Agnesa co la bbròcca")
        == "fra Luca incontrò Agnesa co la bbròcca"
    )


def test_strip_rhyme_metadata_leaves_plain_verse_untouched():
    verse = "Te lo saressi creso, eh Gurgumella,"
    assert strip_rhyme_metadata(verse) == verse


def test_is_valid_hendecasyllable_ignores_accented_metadata():
    line = "ma nun vojo piú affríggeme nun vojjo"
    assert is_valid_hendecasyllable(line, strict=True) is True
    # the leaked suffix must not be counted toward the syllable total
    assert is_valid_hendecasyllable(f"à | {line}", strict=True) is True
    assert is_valid_hendecasyllable(f"ojjo | {line}", strict=True) is True
