import pytest

from common.rhyme_utils import extract_rhyme_suffix, rhyme_key, split_rhyme_metadata


def test_rhyme_key_collapses_accent_variants():
    assert rhyme_key("perché") == rhyme_key("ddischè")  # é vs è -> "e"
    assert rhyme_key("città") == rhyme_key("papà")  # tronche -à -> "a"
    assert rhyme_key("casa") == rhyme_key("rimasa")  # plain branch -> "asa"


def test_rhyme_key_keeps_distinct_rhymes_distinct():
    assert rhyme_key("casa") != rhyme_key("mare")
    assert rhyme_key("città") != rhyme_key("casa")
    assert rhyme_key("amore") != rhyme_key("cantò")


@pytest.mark.parametrize(
    "line, expected",
    [
        # normal tag
        ("<RHYME_A> ella | Te lo saressi", ("<RHYME_A>", "ella", "Te lo saressi")),
        # accented suffix
        (
            "<RHYME_B> à | che mmanco ha ffiato",
            ("<RHYME_B>", "à", "che mmanco ha ffiato"),
        ),
        # single letter tag
        ("Ⓓ raggione | Da per dio!", ("Ⓓ", "raggione", "Da per dio!")),
        # spacing collapsed
        ("<RHYME_A>ella|Te lo saressi", ("<RHYME_A>", "ella", "Te lo saressi")),
        # tag already stripped
        ("ella | Te lo saressi", (None, "ella", "Te lo saressi")),
        # tag-only dataset
        ("<RHYME_A> verse senza pipe", ("<RHYME_A>", None, "verse senza pipe")),
        ("Ⓐ verse senza pipe", ("Ⓐ", None, "verse senza pipe")),
        # no metadata
        (
            "Te lo saressi creso, eh Gurgumella,",
            (None, None, "Te lo saressi creso, eh Gurgumella,"),
        ),
    ],
)
def test_split_rhyme_metadata(line, expected):
    assert split_rhyme_metadata(line) == expected


def test_extract_rhyme_suffix():
    assert extract_rhyme_suffix("Supplisce!!!") == "isce"
    assert extract_rhyme_suffix("Sgraffignar!!") == "ignar"
    assert extract_rhyme_suffix("A") == "a"
    assert extract_rhyme_suffix("Pietà!") == "à"
    assert extract_rhyme_suffix("amor") == "amor"
    assert extract_rhyme_suffix("amore...") == "ore"
    assert extract_rhyme_suffix("rhythms") == "hms"
    assert extract_rhyme_suffix("sciampanella") == "ella"
