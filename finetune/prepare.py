import json
import os
import random
import re
from pathlib import Path

import click

STANZA_SEP = "\n\n<STANZA>\n\n"


def truncate_caudato(text: str) -> tuple[str, bool]:
    """Truncate a sonetto caudato to its first 4 stanzas (octave + sestet), dropping the coda"""
    stanzas = re.split(r"\n\s*\n(?:<STANZA>\n\s*\n)?", text)
    if len(stanzas) <= 4:
        return text, False
    truncated = STANZA_SEP.join(stanzas[:4]).rstrip()
    if not truncated.endswith("<END>"):
        truncated += "\n<END>"
    return truncated, True


def to_natural(text: str) -> str:
    text = re.sub(r"<RHYME_([A-G])>\s*", r"[\1] ", text)
    text = text.replace("<TITLE>", "TITLE: ")
    text = re.sub(r"<SONNET>\s*", "SONNET\n\n", text)
    text = re.sub(r"\s*<STANZA>\s*", "\n\n", text)
    text = re.sub(r"\s*<END>\s*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def write_jsonl(path: Path, items: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path, file_okay=False, exists=True),
    default=Path("data/processed/sonnets/"),
    show_default=True,
    help="Directory containing source sonnet .txt files.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("finetune/data"),
    show_default=True,
    help="Directory where processed output is written.",
)
@click.option(
    "--truncate-caudati",
    is_flag=True,
    help="Truncate sonetti caudati to the first 4 stanzas, dropping coda tercets.",
)
@click.option(
    "--eval-split",
    type=click.FloatRange(0.0, 1.0, max_open=True),
    default=0.05,
    show_default=True,
    help="Fraction held out as eval.",
)
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed.")
def main(
    data_dir: Path,
    output_dir: Path,
    truncate_caudati: bool,
    eval_split: float,
    seed: int,
):
    output_dir = output_dir / data_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    filenames = [fn for fn in os.listdir(data_dir) if fn.endswith(".txt")]

    jsonl_data = []
    truncated_count = 0

    for filename in filenames:
        with open(data_dir / filename, "r", encoding="utf-8") as f:
            text = f.read().strip()
            title = re.search(r"\d{1,}\.\s(.*)\.", filename).group(1).strip("][»")  # type: ignore

            if truncate_caudati:
                text, truncated = truncate_caudato(text)
                if truncated:
                    truncated_count += 1
                    print(f"Truncated coda of {filename}")

            record = "<TITLE>" + title + "\n\n" + text
            record = to_natural(record)
            jsonl_data.append({"text": record})

    summary = f"Processed {len(jsonl_data)} sonnets" + (
        f", {truncated_count} caudati truncated" if truncate_caudati else ""
    )

    if eval_split > 0:
        random.Random(seed).shuffle(jsonl_data)
        n_eval = max(1, round(len(jsonl_data) * eval_split))
        write_jsonl(output_dir / "finetune_eval.jsonl", jsonl_data[:n_eval])
        write_jsonl(output_dir / "finetune_train.jsonl", jsonl_data[n_eval:])
        print(
            f"{summary}: {len(jsonl_data) - n_eval} train / {n_eval} eval "
            f"saved to {output_dir}/finetune_{{train,eval}}.jsonl."
        )
    else:
        write_jsonl(output_dir / "finetune.jsonl", jsonl_data)
        print(f"{summary} and saved to {output_dir / 'finetune.jsonl'}.")


if __name__ == "__main__":
    main()
