import json
from pathlib import Path

import pytest

from common import tokenizer as tokenizer_module
from common.tokenizer import (
    CharTokenizer,
    SyllableTokenizer,
    UnigramTokenizer,
    load_tokenizer,
)


def test_char_tokenizer_encode_decode_with_special_tokens(tmp_path):
    vocab = {
        "special_tokens": {"BOS": "<SONNET>", "EOS": "<END>"},
        "itos": {
            "0": "<SONNET>",
            "1": "a",
            "2": "b",
            "3": "<END>",
        },
        "stoi": {
            "a": 1,
            "b": 2,
        },
    }

    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

    tokenizer = CharTokenizer(str(vocab_path))

    assert tokenizer.encode("ab") == [1, 2]
    assert (
        tokenizer.decode([0, 1, 2, 3], skip_special_tokens=False) == "<SONNET>ab<END>"
    )
    assert tokenizer.decode([0, 1, 2, 3], skip_special_tokens=True) == "ab"


def test_char_tokenizer_decode_strips_rhyme_prefix_metadata(tmp_path):
    vocab = {
        "special_tokens": {"RHYME_D": "Ⓓ"},
        "itos": {
            "0": "Ⓓ",
            "1": " ",
            "2": "r",
            "3": "a",
            "4": "g",
            "5": "i",
            "6": "o",
            "7": "n",
            "8": "e",
            "9": "|",
            "10": "D",
            "11": "p",
            "12": "!",
            "13": "c",
            "14": "’",
            "15": "b",
            "16": "f",
            "17": "t",
            "18": "l",
            "19": "d",
        },
        "stoi": {
            "Ⓓ": 0,
            " ": 1,
            "r": 2,
            "a": 3,
            "g": 4,
            "i": 5,
            "o": 6,
            "n": 7,
            "e": 8,
            "|": 9,
            "D": 10,
            "p": 11,
            "!": 12,
            "c": 13,
            "’": 14,
            "b": 15,
            "f": 16,
            "t": 17,
            "l": 18,
            "d": 19,
        },
    }

    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

    tokenizer = CharTokenizer(str(vocab_path))
    text = "Ⓓ raggione | Da per dio! c’abbi fatto la raggione"
    encoded = tokenizer.encode(text)

    assert tokenizer.decode(encoded, skip_special_tokens=True) == (
        "Da per dio! c’abbi fatto la raggione"
    )
    assert tokenizer.decode(encoded, skip_special_tokens=False) == text


def test_char_tokenizer_decode_preserves_newlines_when_stripping_rhyme_prefix(
    tmp_path,
):
    vocab = {
        "special_tokens": {"RHYME_A": "Ⓐ", "SEP": "\n\n", "EOS": "§"},
        "itos": {
            "0": "Ⓐ",
            "1": " ",
            "2": "c",
            "3": "a",
            "4": "s",
            "5": "|",
            "6": "\n",
            "7": "P",
            "8": "r",
            "9": "i",
            "10": "m",
            "11": "l",
            "12": "n",
            "13": "e",
            "14": "S",
            "15": "o",
            "16": "d",
            "17": "§",
        },
        "stoi": {
            "Ⓐ": 0,
            " ": 1,
            "c": 2,
            "a": 3,
            "s": 4,
            "|": 5,
            "\n": 6,
            "P": 7,
            "r": 8,
            "i": 9,
            "m": 10,
            "l": 11,
            "n": 12,
            "e": 13,
            "S": 14,
            "o": 15,
            "d": 16,
            "§": 17,
        },
    }

    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

    tokenizer = CharTokenizer(str(vocab_path))
    text = "Ⓐ casa | Prima\n\nⒶ casa | Seconda§"
    encoded = tokenizer.encode(text)

    assert tokenizer.decode(encoded, skip_special_tokens=True) == "Prima\n\nSeconda"


def test_syllable_tokenizer_decode_strips_rhyme_prefix(tmp_path):
    pieces = [
        "<RHYME_A>",
        "<RHYME_B>",
        "<STANZA>",
        "ella",
        "à",
        "|",
        "Te",
        " lo",
        "che",
        " annà",
        "\n",
    ]
    itos = {str(i): p for i, p in enumerate(pieces)}
    stoi = {p: i for i, p in enumerate(pieces)}
    vocab = {
        "special_tokens": {
            "RHYME_A": "<RHYME_A>",
            "RHYME_B": "<RHYME_B>",
            "SEP": "<STANZA>",
        },
        "itos": itos,
        "stoi": stoi,
    }
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

    tokenizer = SyllableTokenizer(str(vocab_path))
    # "<RHYME_A>ella|Te lo\n<RHYME_B>à|che annà\n<STANZA>"
    ids = [
        stoi[p]
        for p in [
            "<RHYME_A>",
            "ella",
            "|",
            "Te",
            " lo",
            "\n",
            "<RHYME_B>",
            "à",
            "|",
            "che",
            " annà",
            "\n",
            "<STANZA>",
        ]
    ]

    assert tokenizer.decode(ids, skip_special_tokens=True) == (
        "Te lo\nche annà\n<STANZA>"
    )


def test_unigram_flatten_ids_handles_nested_and_flat_lists():
    tokenizer = UnigramTokenizer.__new__(UnigramTokenizer)

    assert tokenizer._flatten_ids([[1, 2], [3]]) == [1, 2, 3]
    assert tokenizer._flatten_ids([4, 5, 6]) == [4, 5, 6]


class _DummyCharTokenizer:
    def __init__(self, path):
        self.kind = "char"
        self.path = path


class _DummyBPETokenizer:
    def __init__(self, path):
        self.kind = "bpe"
        self.path = path


class _DummyUnigramTokenizer:
    def __init__(self, path):
        self.kind = "unigram"
        self.path = path


def test_load_tokenizer_dispatches_by_explicit_type(monkeypatch):
    monkeypatch.setattr(tokenizer_module, "CharTokenizer", _DummyCharTokenizer)
    monkeypatch.setattr(tokenizer_module, "BPETokenizer", _DummyBPETokenizer)
    monkeypatch.setattr(tokenizer_module, "UnigramTokenizer", _DummyUnigramTokenizer)

    char_tok = load_tokenizer(Path("foo/vocab.json"), "char")
    bpe_tok = load_tokenizer(Path("foo/tokenizer.json"), "bpe")
    unigram_tok = load_tokenizer(Path("foo/model.model"), "unigram")

    assert char_tok.kind == "char"
    assert bpe_tok.kind == "bpe"
    assert unigram_tok.kind == "unigram"
    assert char_tok.path == "foo/vocab.json"
    assert bpe_tok.path == "foo/tokenizer.json"
    assert unigram_tok.path == "foo/model.model"


def test_load_tokenizer_infers_from_path_when_type_unknown(monkeypatch):
    monkeypatch.setattr(tokenizer_module, "CharTokenizer", _DummyCharTokenizer)
    monkeypatch.setattr(tokenizer_module, "BPETokenizer", _DummyBPETokenizer)
    monkeypatch.setattr(tokenizer_module, "UnigramTokenizer", _DummyUnigramTokenizer)

    inferred_char = load_tokenizer(Path("foo/vocab.json"), "auto")
    inferred_unigram = load_tokenizer(Path("foo/tokenizer.model"), "auto")
    inferred_bpe = load_tokenizer(Path("foo/tokenizer.json"), "auto")

    assert inferred_char.kind == "char"
    assert inferred_unigram.kind == "unigram"
    assert inferred_bpe.kind == "bpe"


def test_load_tokenizer_invalid_type():
    with pytest.raises(ValueError, match="Invalid tokenizer_type"):
        load_tokenizer(Path("foo/tokenizer.json"), "wordpiece")
