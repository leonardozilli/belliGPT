import click

from common.eval import evaluate_structure
from common.report import aggregate, print_report
from common.rhyme_utils import rhyme_key
from finetune import generate as _gen
from finetune_constrained import decode as _dec
from finetune_constrained import title_bias as _tb
from finetune_constrained.lexicon import build_lexicon


def schema_adherence(verses: list[str], scheme: str) -> tuple[int, int]:
    by_letter: dict[str, list[str]] = {}
    for letter, verse in zip(scheme, verses):
        words = verse.split()
        if words:
            by_letter.setdefault(letter, []).append(rhyme_key(words[-1]))
    hit = tot = 0
    for keys in by_letter.values():
        if len(keys) < 2:
            continue
        majority = max(set(keys), key=keys.count)
        hit += sum(1 for k in keys if k == majority)
        tot += len(keys)
    return hit, tot


@click.command()
@click.option("--adapter", "adapter_dir", default=None, help="Adapter directory.")
@click.option(
    "--model", "model_group", default=None, help="model/ group (if no --adapter)."
)
@click.option(
    "--dataset", "dataset_group", default=None, help="dataset/ group (if no --adapter)."
)
@click.option(
    "--backend",
    type=click.Choice(["unsloth", "llamacpp"]),
    default="unsloth",
    show_default=True,
    help="Decoding backend: unsloth (HF adapter) or llamacpp (GGUF base+LoRA).",
)
@click.option(
    "--gguf",
    "gguf_path",
    default=None,
    help="[llamacpp] GGUF base path.",
)
@click.option(
    "--lora-gguf",
    "lora_gguf",
    default=None,
    help="[llamacpp] GGUF LoRA path; pass '' to run the bare gguf base).",
)
@click.option(
    "--lexicon-dir",
    default="data/processed/sonnets_rhymes",
    show_default=True,
    help="Corpus to build the rhyme lexicon from.",
)
@click.option(
    "--title",
    default=None,
    help="Sonnet title to prompt with; omit for free generation.",
)
@click.option("-n", "--num-samples", default=5, show_default=True)
@click.option("-t", "--temperature", default=0.8, show_default=True)
@click.option("--top-p", default=0.95, show_default=True)
@click.option(
    "--repetition-penalty",
    default=1.3,
    show_default=True,
    help="Repetition penalty.",
)
@click.option(
    "--title-bias",
    default=0.0,
    show_default=True,
    help="Weight on title relevance in rhyme-word choice: pick = z(logprob)+λ·z(sim). Implies --title.",
)
@click.option(
    "--relevance",
    type=click.Choice(["embed", "encoder"]),
    default="embed",
    show_default=True,
    help="Title-relevance source: embed (model's input-embedding table) or encoder (sentence-transformer). Implies --title-bias > 0.",
)
@click.option(
    "--encoder-model",
    default=_tb.DEFAULT_ENCODER,
    show_default=True,
    help="[--relevance encoder] sentence-transformer model name.",
)
@click.option(
    "--scheme",
    default=_dec.DEFAULT_SCHEME,
    show_default=True,
    help=f"Rhyme scheme to use (default {_dec.DEFAULT_SCHEME}).",
)
@click.option(
    "--min-class-size",
    default=5,
    show_default=True,
    help="Minimum number of words in a rhyme class.",
)
@click.option(
    "--strict-meter", is_flag=True, help="Require 11 syllables in [lo,hi] (else 10-12)."
)
@click.option("--seed", default=0, show_default=True)
def main(
    adapter_dir,
    model_group,
    dataset_group,
    backend,
    gguf_path,
    lora_gguf,
    lexicon_dir,
    title,
    num_samples,
    temperature,
    top_p,
    repetition_penalty,
    title_bias,
    relevance,
    encoder_model,
    scheme,
    min_class_size,
    strict_meter,
    seed,
):
    import torch

    click.echo(f"[constrained] building lexicon from {lexicon_dir}")
    lex = build_lexicon(lexicon_dir)
    click.echo(
        f"[constrained] {len(lex.by_key)} rhyme classes, "
        f"{sum(len(v) for v in lex.by_key.values())} words"
    )

    dcfg = _dec.ConstrainedConfig(
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        min_class_size=min_class_size,
        strict_meter=strict_meter,
        title_bias=title_bias,
    )

    if backend == "llamacpp":
        from finetune import llamacpp as _llm
        from finetune_constrained.backend import LlamaCppBackend

        gguf = gguf_path or _llm.DEFAULT_GGUF
        lora = _llm.DEFAULT_LORA if lora_gguf is None else (lora_gguf or None)
        click.echo(
            f"[constrained] backend=llamacpp gguf={gguf} lora={lora or '(none)'}"
        )
        engine = _llm.load_engine(gguf, lora, logits_all=True)
        be = LlamaCppBackend(engine)
    else:
        from finetune_constrained.backend import HFBackend

        if adapter_dir is None:
            from finetune.config import FinetuneConfig

            cfg = FinetuneConfig.compose(
                group_overrides={"model": model_group, "dataset": dataset_group}
            )
            adapter_dir = _gen.adapter_dir_for(cfg)
        click.echo(f"[constrained] backend=unsloth loading adapter: {adapter_dir}")
        model, tok = _gen.load_adapter(adapter_dir)
        be = HFBackend(model, tok, chunk=dcfg.chunk)

    prompt = _gen.build_prompt(title)

    relevance_fn = None
    if title_bias > 0:
        if not title:
            raise click.UsageError("--title-bias needs a --title to be relevant to.")
        words = [w.text for ws in lex.by_key.values() for w in ws]
        if relevance == "encoder":
            click.echo(
                f"[constrained] encoding lexicon for title bias via {encoder_model}"
            )
            tb = _tb.EncoderTitleBias(words, model_name=encoder_model)
        else:
            click.echo("[constrained] precomputing lexicon embeddings for title bias")
            tb = _tb.TitleBias(be, words)
        tb.set_title(title)
        relevance_fn = tb.relevance

    samples = []
    adh_hit = adh_tot = 0
    for i in range(num_samples):
        torch.manual_seed(seed + i)
        out = _dec.generate_sonnet(
            be, lex, prompt, scheme=scheme, cfg=dcfg, relevance_fn=relevance_fn
        )
        res = evaluate_structure(out["eval_text"], strict=strict_meter)
        h, t = schema_adherence(out["verses"], scheme)
        adh_hit += h
        adh_tot += t
        samples.append(
            {
                "index": i,
                "truncated": False,  # lines are forced, so truncation can't occur
                "empty": res["line_count"] == 0,
                "overlap": None,
                "text": out["marked"],
                "metrics": res,
            }
        )

        click.echo(f"\n{'=' * 64}\nsample {i + 1}/{num_samples}\n{'=' * 64}")
        click.echo(out["marked"])
        click.echo(
            f"\n[eval] 14_lines={res['is_14_lines']} "
            f"structure={res['is_correct_structure']} "
            f"hendec={res['valid_hendecasyllables']}/{res['line_count']} "
            f"rhyme_meter={res['is_valid_rhyme_meter']} "
            f"valid_sonnet={res['is_valid_sonnet']}  "
            f"rhyme adherence={h}/{t}"
        )

    report = aggregate(samples, None)
    print_report(report, strict_meter)


if __name__ == "__main__":
    main()
