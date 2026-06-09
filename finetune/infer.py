import re

import click

from finetune import generate as _gen
from finetune.config import FinetuneConfig

_RHYME_PREFIX = re.compile(r"(?m)^[ \t]*\[[A-G]\][ \t]*")
_HEADER_LINE = re.compile(r"(?m)^[ \t]*(?:TITLE:.*|SONNET[ \t]*)$")


def _for_structure_eval(text: str) -> str:
    """Strip the title/sonnet header lines and the rhyme markers, leaving just the raw text for evaluation."""
    text = _HEADER_LINE.sub("", text)
    text = _RHYME_PREFIX.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Base config file.",
)
@click.option(
    "--model",
    "model_group",
    default=None,
    help="Model group choice (must match the trained adapter).",
)
@click.option(
    "--dataset",
    "dataset_group",
    default=None,
    help="Dataset group choice (must match the trained adapter).",
)
@click.option(
    "--adapter",
    "adapter_dir",
    default=None,
    help="Adapter directory (overrides --config/--model/--dataset).",
)
@click.option(
    "--title",
    default=None,
    help="Sonnet title to prompt with. Omit for free generation (model "
    "invents the title); pass '' for SONNET-only.",
)
@click.option("-n", "--num-samples", default=3, show_default=True)
@click.option("-t", "--temperature", default=0.8, show_default=True)
@click.option("--top-p", default=0.95, show_default=True)
@click.option(
    "--repetition-penalty",
    default=1.0,
    show_default=True,
    help="Repetition penalty.",
)
@click.option("--max-new-tokens", default=400, show_default=True)
@click.option(
    "--strip-tags",
    is_flag=True,
    help="Strip the rhyme markers from the generated text.",
)
@click.option("--eval", "do_eval", is_flag=True, help="Score generation.")
@click.option(
    "--strict",
    is_flag=True,
    help="Strict meter eval: only count lines with exactly 11 syllables as valid.",
)
@click.option(
    "--train-corpus",
    type=click.Path(exists=True),
    default=None,
    help="Training corpus (.txt file or dir) for the overlap / memorization check "
    "(implies --eval).",
)
@click.option("--overlap-n", default=4, show_default=True, help="Overlap n-gram size.")
@click.option(
    "--json-out",
    type=click.Path(),
    default=None,
    help="Write results to a JSON file.",
)
def main(
    config_path,
    model_group,
    dataset_group,
    adapter_dir,
    title,
    num_samples,
    temperature,
    top_p,
    repetition_penalty,
    max_new_tokens,
    strip_tags,
    do_eval,
    strict,
    train_corpus,
    overlap_n,
    json_out,
):
    if adapter_dir is None:
        cfg = FinetuneConfig.compose(
            base=config_path,
            group_overrides={"model": model_group, "dataset": dataset_group},
        )
        adapter_dir = _gen.adapter_dir_for(cfg)
    click.echo(f"[infer] loading adapter: {adapter_dir}")

    model, tokenizer = _gen.load_adapter(adapter_dir)
    prompt = _gen.build_prompt(title)

    do_score = do_eval or train_corpus is not None
    corpus_ngrams = None
    if do_score:
        from common.eval import evaluate_structure, ngram_overlap
        from common.report import aggregate, load_corpus_ngrams, print_report

        if train_corpus is not None:
            corpus_ngrams = load_corpus_ngrams(train_corpus, overlap_n)
            click.echo(
                f"[infer] loaded {len(corpus_ngrams)} reference {overlap_n}-grams "
                f"from {train_corpus}"
            )

    samples = []
    for i in range(num_samples):
        click.echo(f"\n{'=' * 60}\nsample {i + 1}/{num_samples}\n{'=' * 60}")
        text = _gen.generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stream=False,
            strip_tags=strip_tags,
        )
        click.echo(text)
        if do_score:
            eval_text = _for_structure_eval(text)
            res = evaluate_structure(eval_text, strict=strict)
            overlap = None
            if corpus_ngrams is not None:
                overlap = ngram_overlap(eval_text, corpus_ngrams, overlap_n)
            samples.append(
                {
                    "index": i,
                    "truncated": False,
                    "empty": res["line_count"] == 0,
                    "overlap": overlap,
                    "text": text,
                    "metrics": res,
                }
            )
            click.echo(
                f"\n[eval] 14_lines={res['is_14_lines']} "
                f"stanzas={res['valid_stanzas']}/4 "
                f"hendec={res['valid_hendecasyllables']}/{res['line_count']} "
                f"rhyme_lines={res['rhyme_lines']} "
                f"valid_sonnet={res['is_valid_sonnet']}"
            )

    if do_score:
        report = aggregate(samples, overlap_n if corpus_ngrams is not None else None)
        print_report(report, strict)
        if json_out is not None:
            import json

            payload = {
                "params": {
                    "adapter": adapter_dir,
                    "title": title,
                    "num_samples": num_samples,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "max_new_tokens": max_new_tokens,
                    "strict": strict,
                    "train_corpus": train_corpus,
                    "overlap_n": overlap_n if corpus_ngrams is not None else None,
                },
                "aggregates": report,
                "samples": samples,
            }
            with open(json_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            click.echo(f"\nWrote results to {json_out}")


if __name__ == "__main__":
    main()
