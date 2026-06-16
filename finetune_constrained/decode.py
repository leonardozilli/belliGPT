from dataclasses import dataclass

from common.eval import _range_is_valid
from finetune_constrained.lexicon import Lexicon, Word
from syllable.syllabify import count_syllable_range

DEFAULT_SCHEME = "ABBAABBA" + "CDCDCD"
_STANZA_FIRST = {4, 8, 11}
_TARGET_SYLL = 11


@dataclass
class ConstrainedConfig:
    temperature: float = 0.8
    top_p: float = 0.95
    min_body_syll: int = 6
    min_class_size: int = 5
    defining_cap: int = 80
    class_cap: int = 80
    score_cap: int = 60
    chunk: int = 12
    max_body_tokens: int = 28
    strict_meter: bool = False
    repetition_penalty: float = 1.3
    rep_window: int = 100
    no_repeat_ngram: int = 3
    title_bias: float = 0.0
    defining_cap_relevance: int = 400


def _word_tokens(lex: Lexicon, backend, word: str) -> list[int]:
    cached = lex._tok_cache.get(word)
    if cached is None:
        cached = backend.encode_word(word)
        lex._tok_cache[word] = cached
    return cached


def _apply_penalties(logits, context: list[int], cfg: "ConstrainedConfig"):
    """Windowed repetition penalty (CTRL-style), n-gram ban."""
    import torch

    if cfg.repetition_penalty and cfg.repetition_penalty != 1.0:
        window = context[-cfg.rep_window :] if cfg.rep_window else context
        seen = sorted(set(window))
        if seen:
            idx = torch.tensor(seen, device=logits.device)
            v = logits[idx]
            logits[idx] = torch.where(
                v > 0, v / cfg.repetition_penalty, v * cfg.repetition_penalty
            )
    n = cfg.no_repeat_ngram
    if n and n >= 2 and len(context) >= n - 1:
        prefix = tuple(context[-(n - 1) :])
        banned = {
            context[i + n - 1]
            for i in range(len(context) - n + 1)
            if tuple(context[i : i + n - 1]) == prefix
        }
        if banned:
            logits[torch.tensor(sorted(banned), device=logits.device)] = float("-inf")
    return logits


def _sample(logits, temperature: float, top_p: float) -> int:
    """Temperature + nucleus sampling."""
    import torch

    if temperature <= 0:
        return int(torch.argmax(logits))
    probs = torch.softmax(logits.float() / max(temperature, 1e-6), dim=-1)

    sp, si = torch.sort(probs, descending=True)
    cum = torch.cumsum(sp, dim=-1)
    keep = (cum - sp) < top_p
    sp = torch.where(keep, sp, torch.zeros_like(sp))
    if float(sp.sum()) <= 0:
        sp = probs.clone()
    sp = sp / sp.sum()
    choice = int(torch.multinomial(sp, 1))
    return int(si[choice])


def _line_range(body_text: str, word: str) -> tuple[int, int]:
    """Returns the min and max syllables possible in the line if we add this word"""
    return count_syllable_range((body_text + " " + word).strip())


def _standardize(xs: list[float]) -> list[float]:
    """Zero-mean unit std"""
    n = len(xs)
    if n == 0:
        return xs
    mean = sum(xs) / n
    sd = (sum((x - mean) ** 2 for x in xs) / n) ** 0.5
    if sd < 1e-9:
        return [0.0] * n
    return [(x - mean) / sd for x in xs]


def _choose_word(
    backend,
    lex: Lexicon,
    context: list[int],
    body_text: str,
    target: str | None,
    used: set[str],
    used_words: set[str],
    cfg: ConstrainedConfig,
    strict: bool,
    relevance_fn=None,
) -> Word | None:
    use_rel = relevance_fn is not None and cfg.title_bias > 0
    if target is not None:
        words = lex.by_key.get(target, [])[: cfg.class_cap]
    else:
        cap = cfg.defining_cap_relevance if use_rel else cfg.defining_cap
        words = lex.candidates_pool(cfg.min_class_size, used, cap)
    fresh = [w for w in words if w.text.lower() not in used_words]
    words = fresh or words
    if not words:
        return None

    fits = []
    for w in words:
        lo, hi = _line_range(body_text, w.text)
        ok = (lo <= 11 <= hi) if cfg.strict_meter else _range_is_valid(lo, hi, False)
        if ok:
            fits.append(w)

    if not fits:
        if strict:
            return None

        def _dist(w: Word) -> float:
            lo, hi = _line_range(body_text, w.text)
            return abs((lo + hi) / 2 - _TARGET_SYLL)

        # sort words to choose the ones that get us as close as possible to 11 sillables
        fits = sorted(words, key=_dist)

    if use_rel:
        rel_all = relevance_fn([w.text for w in fits])
        order = sorted(range(len(fits)), key=lambda i: rel_all[i], reverse=True)
        fits = [fits[i] for i in order[: cfg.score_cap]]
        rel = [rel_all[i] for i in order[: cfg.score_cap]]
    else:
        fits = fits[: cfg.score_cap]

    toks = [_word_tokens(lex, backend, w.text) for w in fits]
    scores = backend.score_candidates(context, toks)
    if use_rel:
        combined = [
            a + cfg.title_bias * b
            for a, b in zip(_standardize(scores), _standardize(rel))
        ]
    else:
        combined = scores
    return fits[max(range(len(fits)), key=lambda i: combined[i])]


def _gen_line(
    backend,
    lex: Lexicon,
    context: list[int],
    label: str,
    rhyme_map: dict[str, str],
    used: set[str],
    used_words: set[str],
    cfg: ConstrainedConfig,
    relevance_fn=None,
) -> str:
    target = rhyme_map.get(label)
    body: list[int] = []
    special = backend.special_ids

    def finish(w: Word) -> str:
        context.extend(_word_tokens(lex, backend, w.text))
        if target is None:
            rhyme_map[label] = w.key
            used.add(w.key)
        used_words.add(w.text.lower())
        return (backend.decode(body).strip() + " " + w.text).strip()

    while True:
        logits = _apply_penalties(backend.forward_last(context), context, cfg)
        t = _sample(logits, cfg.temperature, cfg.top_p)
        boundary = backend.is_word_start(t)
        want_nl = t in special or "\n" in backend.decode([t])
        body_text = backend.decode(body).strip() if body else ""
        lo = count_syllable_range(body_text)[0] if body_text else 0

        # 1) body is long enough and the model wants to end the line -> pick a rhyme word and finish
        if body and (boundary or want_nl) and lo >= cfg.min_body_syll:
            w = _choose_word(
                backend,
                lex,
                context,
                body_text,
                target,
                used,
                used_words,
                cfg,
                strict=True,
                relevance_fn=relevance_fn,
            )
            if w is not None:
                return finish(w)

        # 2) model wants to end the line but the body is too short -> if it's close
        # to enough syllables, force a relaxed rhyme; otherwise, drop the newline and continue.
        if want_nl:
            if body and lo >= cfg.min_body_syll - 2:
                w = _choose_word(
                    backend,
                    lex,
                    context,
                    body_text,
                    target,
                    used,
                    used_words,
                    cfg,
                    strict=False,
                    relevance_fn=relevance_fn,
                )
                if w is not None:
                    return finish(w)
            continue  # too short: drop the newline, keep writing the body

        # 3) line is about to exceed target syllables -> force a rhyme word and finish
        if body and (lo >= _TARGET_SYLL + 1 or len(body) >= cfg.max_body_tokens):
            w = _choose_word(
                backend,
                lex,
                context,
                body_text,
                target,
                used,
                used_words,
                cfg,
                strict=False,
                relevance_fn=relevance_fn,
            )
            if w is not None:
                return finish(w)

        # 4) otherwise, accept the token and keep going
        body.append(t)
        context.append(t)


def generate_sonnet(
    backend,
    lex: Lexicon,
    prompt: str,
    scheme: str = DEFAULT_SCHEME,
    cfg: ConstrainedConfig | None = None,
    relevance_fn=None,
) -> dict:
    """Generate one constrained sonnet."""
    cfg = cfg or ConstrainedConfig()
    rhyme_map: dict[str, str] = {}
    used: set[str] = set()
    used_words: set[str] = set()
    verses: list[str] = []
    text = prompt  # "TITLE: x\n\nSONNET\n\n"

    for i, label in enumerate(scheme):
        sep = "" if i == 0 else ("\n\n" if i in _STANZA_FIRST else "\n")
        text += f"{sep}[{label}]"
        context = backend.encode(text, add_special=True)
        verse = _gen_line(
            backend,
            lex,
            context,
            label,
            rhyme_map,
            used,
            used_words,
            cfg,
            relevance_fn,
        )
        verses.append(verse)
        text += " " + verse

    stanzas = [verses[0:4], verses[4:8], verses[8:11], verses[11:14]]
    eval_text = "\n\n".join("\n".join(s) for s in stanzas)
    marked_stanzas = [
        [f"[{scheme[idx]}] {v}" for idx, v in zip(rng, s)]
        for rng, s in zip(
            [range(0, 4), range(4, 8), range(8, 11), range(11, 14)], stanzas
        )
    ]
    marked = "\n\n".join("\n".join(s) for s in marked_stanzas)
    return {"marked": marked, "verses": verses, "eval_text": eval_text}
