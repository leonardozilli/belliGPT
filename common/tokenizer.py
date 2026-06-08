import json
from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer

from common.rhyme_utils import split_rhyme_metadata
from syllable.syllabify import syllabify_text


def _strip_rhyme_prefix(text: str) -> str:
    "Remove the leading ``<tag> <suffix> |`` rhyme prefix from each line."
    out = []
    for line in text.split("\n"):
        tag, suffix, verse = split_rhyme_metadata(line)
        if tag is not None and suffix is not None:
            out.append(verse)
        else:
            out.append(line)
    return "\n".join(out)


SPECIAL_TOKENS = {
    "BOS": "<SONNET>",
    "SEP": "<STANZA>",
    "EOS": "<END>",
    "RHYME_A": "<RHYME_A>",
    "RHYME_B": "<RHYME_B>",
    "RHYME_C": "<RHYME_C>",
    "RHYME_D": "<RHYME_D>",
    "RHYME_E": "<RHYME_E>",
    "RHYME_F": "<RHYME_F>",
    "RHYME_G": "<RHYME_G>",
}


class CharTokenizer:
    def __init__(self, path: str | None = None):
        self.kind = "char"
        self.path = path
        self.special_tokens = {}
        self.itos = {}
        self.stoi = {}
        self.vocab_size = 0

        if path:
            with open(path, "r") as f:
                vocab = json.load(f)

            self.special_tokens = vocab["special_tokens"]
            self.itos = {int(k): v for k, v in vocab["itos"].items()}
            self.stoi = vocab["stoi"]
            self.vocab_size = len(self.itos)

    def encode(self, text: str):
        return [self.stoi[c] for c in text]

    def get_token_id(self, token: str) -> int:
        return self.stoi.get(
            token, self.stoi.get(self.special_tokens.get("UNK", ""), -1)
        )

    def decode(self, batch: list, skip_special_tokens: bool = True):
        if isinstance(batch, int):
            batch = [batch]

        text = "".join(self.itos[i] for i in batch)

        if skip_special_tokens:
            text = _strip_rhyme_prefix(text)
            tokens_to_strip = {
                token for name, token in self.special_tokens.items() if name != "SEP"
            }
            for token in tokens_to_strip:
                text = text.replace(token, "")

            return text

        return text


class SyllableTokenizer:
    def __init__(self, path: str | None = None):
        self.kind = "syllable"
        self.path = path
        self.special_tokens = {}
        self.itos = {}
        self.stoi = {}
        self.vocab_size = 0

        if path:
            with open(path, "r") as f:
                vocab = json.load(f)

            self.special_tokens = vocab["special_tokens"]
            self.itos = {int(k): v for k, v in vocab["itos"].items()}
            self.stoi = vocab["stoi"]
            self.vocab_size = len(self.itos)

    def encode(self, text: str):
        return [self.stoi[syl] for syl in syllabify_text(text) if syl in self.stoi]

    def get_token_id(self, token: str) -> int:
        return self.stoi.get(
            token, self.stoi.get(self.special_tokens.get("UNK", ""), -1)
        )

    def decode(self, batch: list, skip_special_tokens: bool = True):
        if isinstance(batch, int):
            batch = [batch]

        text = "".join(self.itos[i] for i in batch)

        if skip_special_tokens:
            text = _strip_rhyme_prefix(text)
            tokens_to_strip = {
                token for name, token in self.special_tokens.items() if name != "SEP"
            }
            for token in tokens_to_strip:
                text = text.replace(token, "")

            return text

        return text


class BPETokenizer:
    def __init__(self, path: str):
        self.kind = "bpe"
        self.path = path
        self.tokenizer = Tokenizer.from_file(path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.special_tokens = SPECIAL_TOKENS.copy()

    def encode(self, text: str):
        return self.tokenizer.encode(text).ids

    def get_token_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def decode(self, batch: list, skip_special_tokens: bool = True):
        return self.tokenizer.decode(batch, skip_special_tokens=skip_special_tokens)


class UnigramTokenizer:
    def __init__(self, path: str):
        self.kind = "unigram"
        self.path = path

        self.special_tokens = SPECIAL_TOKENS.copy()
        self.special_tokens.update({"UNK": "<UNK>"})

        if path.endswith(".model"):
            import sentencepiece as spm

            spm.set_min_log_level(2)  # suppress warnings

            self.backend = "sentencepiece"
            self.kind = "unigram_regularized"
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(path)
            self.vocab_size = self.sp.vocab_size()
        else:
            self.backend = "hf"
            self.tokenizer = Tokenizer.from_file(path)
            self.vocab_size = self.tokenizer.get_vocab_size()

    def _flatten_ids(self, batch: Iterable[int] | Iterable[Iterable[int]]):
        ids = list(batch)
        if ids and isinstance(ids[0], list):
            return [item for sublist in ids for item in sublist]
        return ids

    def encode(self, text: str):
        if self.backend == "sentencepiece":
            return self.sp.EncodeAsIds(text)
        return self.tokenizer.encode(text).ids

    def get_token_id(self, token: str) -> int:
        if self.backend == "sentencepiece":
            return self.sp.PieceToId(token)
        return self.tokenizer.token_to_id(token)

    def decode(self, batch: list, skip_special_tokens: bool = True):
        ids = self._flatten_ids(batch)

        if self.backend == "sentencepiece":
            text = self.sp.Decode(ids)
            if skip_special_tokens:
                for token in self.special_tokens.values():
                    text = text.replace(token, "")
            return text

        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


def load_tokenizer(
    tokenizer_path: Path, tokenizer_type: str
) -> CharTokenizer | SyllableTokenizer | BPETokenizer | UnigramTokenizer:
    tokenizer_path_str = str(tokenizer_path)

    valid_tokenizer_types = {"char", "syllable", "bpe", "unigram", "auto"}
    if tokenizer_type not in valid_tokenizer_types:
        raise ValueError(
            "Invalid tokenizer_type "
            f"{tokenizer_type!r}. Expected one of: {sorted(valid_tokenizer_types)}"
        )

    if tokenizer_type == "char":
        return CharTokenizer(tokenizer_path_str)
    if tokenizer_type == "syllable":
        return SyllableTokenizer(tokenizer_path_str)
    if tokenizer_type == "bpe":
        return BPETokenizer(tokenizer_path_str)
    if tokenizer_type == "unigram":
        return UnigramTokenizer(tokenizer_path_str)

    if tokenizer_path.name == "vocab.json":
        return CharTokenizer(tokenizer_path_str)
    if tokenizer_path.suffix == ".model":
        return UnigramTokenizer(tokenizer_path_str)
    return BPETokenizer(tokenizer_path_str)
