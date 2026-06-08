import json
import random

import click
import torch

from common.eval import evaluate_structure, ngram_overlap
from common.model import GBT
from common.report import aggregate, load_corpus_ngrams, print_report
from common.tokenizer import load_tokenizer


@click.command()
@click.option("--checkpoint", required=True, help="Path to the .pt checkpoint file")
@click.option(
    "--tokenizer-path", required=True, help="Path to vocab.json or spm_model.model"
)
@click.option(
    "--tokenizer-type",
    type=click.Choice(["char", "syllable", "unigram", "bpe"]),
    required=True,
)
@click.option(
    "--num-samples", default=10, help="Number of sonnets to generate and evaluate"
)
@click.option("--temperature", default=0.8, help="Generation temperature")
@click.option("--top-k", default=40, help="Top-K sampling")
@click.option("--top-p", default=0.9, help="Top-P (nucleus) sampling")
@click.option(
    "--strict",
    is_flag=True,
    help="If enabled, only count lines as valid if they have exactly 11 syllables, otherwise 10 and 12 syllables are also counted as valid",
)
@click.option("--silent", is_flag=True, help="Suppress generated text output")
@click.option("--seed", type=int, default=None, help="Random seed")
@click.option(
    "--json-out",
    type=click.Path(),
    default=None,
    help="Write results to this JSON file",
)
@click.option(
    "--train-corpus",
    type=click.Path(exists=True),
    default=None,
    help="Training corpus (.txt file or dir) for the n-gram check",
)
@click.option("--overlap-n", default=4, help="Word n-gram size for the overlap check")
def main(
    checkpoint,
    tokenizer_path,
    tokenizer_type,
    num_samples,
    temperature,
    top_k,
    top_p,
    strict,
    silent,
    seed,
    json_out,
    train_corpus,
    overlap_n,
):
    if num_samples <= 0:
        raise click.BadParameter(
            "must be a positive integer", param_hint="--num-samples"
        )

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GBT.from_pretrained(checkpoint).to(device)
    model.eval()
    tokenizer = load_tokenizer(tokenizer_path, tokenizer_type)

    bos_str = tokenizer.special_tokens.get("BOS", "<SONNET>")
    eos_str = tokenizer.special_tokens.get("EOS", "<END>")

    bos_id = tokenizer.get_token_id(bos_str)
    eos_id = tokenizer.get_token_id(eos_str)

    max_new_tokens = 1024 if tokenizer_type == "char" else 512

    corpus_ngrams = None
    if train_corpus is not None:
        corpus_ngrams = load_corpus_ngrams(train_corpus, overlap_n)
        print(
            f"Loaded {len(corpus_ngrams)} reference {overlap_n}-grams "
            f"from {train_corpus}\n"
        )

    print(f"Evaluating {num_samples} samples...\n")

    samples = []
    with torch.no_grad():
        for i in range(num_samples):
            context = torch.tensor([[bos_id]], dtype=torch.long, device=device)

            pred = model.generate(
                context,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_id=eos_id,
            )

            token_ids = pred[0].tolist()
            # generate() appends eos_id and breaks on EOS, so a run that ends on
            # any other token hit the length cap -> truncated.
            truncated = eos_id is not None and token_ids[-1] != eos_id
            generated_text = tokenizer.decode(token_ids, skip_special_tokens=True)
            full_text = tokenizer.decode(token_ids, skip_special_tokens=False).replace(
                "<NEWLINE>", "\n"
            )

            metrics = evaluate_structure(generated_text, strict=strict)
            overlap = None
            if corpus_ngrams is not None:
                overlap = ngram_overlap(generated_text, corpus_ngrams, overlap_n)

            samples.append(
                {
                    "index": i,
                    "truncated": truncated,
                    "empty": metrics["line_count"] == 0,
                    "overlap": overlap,
                    "text": full_text,
                    "metrics": metrics,
                }
            )

            if not silent:
                print(f"--- Sample {i + 1} ---")
                print(full_text)
                print("----------------\n")

    report = aggregate(samples, overlap_n if corpus_ngrams is not None else None)
    print_report(report, strict)

    if json_out is not None:
        payload = {
            "params": {
                "checkpoint": checkpoint,
                "tokenizer_path": tokenizer_path,
                "tokenizer_type": tokenizer_type,
                "num_samples": num_samples,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "strict": strict,
                "seed": seed,
                "max_new_tokens": max_new_tokens,
                "train_corpus": train_corpus,
                "overlap_n": overlap_n if corpus_ngrams is not None else None,
            },
            "aggregates": report,
            "samples": samples,
        }
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nWrote results to {json_out}")


if __name__ == "__main__":
    main()
