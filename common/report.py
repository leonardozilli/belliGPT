import glob
import os
from collections import Counter

from common.eval import strip_rhyme_metadata, wilson_interval, word_ngrams


def load_corpus_ngrams(path: str, n: int) -> set:
    """Build the reference set of word n-grams from the train corpus."""
    files = (
        [path]
        if os.path.isfile(path)
        else sorted(glob.glob(os.path.join(path, "**/*.txt"), recursive=True))
    )
    reference: set = set()
    for fp in files:
        raw = open(fp, encoding="utf-8").read()
        raw = raw.replace("<NEWLINE>", "\n")
        verse = " ".join(
            strip_rhyme_metadata(line) for line in raw.splitlines() if line.strip()
        )
        reference |= word_ngrams(verse, n)
    return reference


def aggregate(samples, overlap_n):
    evaluated = [s for s in samples if not s["empty"]]
    n_eval = len(evaluated)
    skipped = sum(1 for s in samples if s["empty"])
    truncated = sum(1 for s in samples if s["truncated"])

    per_sonnet_keys = [
        "is_14_lines",
        "is_correct_structure",
        "is_valid_stanzas",
        "is_valid_rhyme_meter",
        "is_valid_sonnet",
    ]
    per_sonnet = {}
    for key in per_sonnet_keys:
        k = sum(1 for s in evaluated if s["metrics"][key])
        low, high = wilson_interval(k, n_eval)
        per_sonnet[key] = {
            "count": k,
            "total": n_eval,
            "rate": (k / n_eval) if n_eval else None,
            "ci95": [low, high],
        }

    total_lines = sum(s["metrics"]["line_count"] for s in evaluated)
    valid_hend = sum(s["metrics"]["valid_hendecasyllables"] for s in evaluated)
    rhyme_lines = sum(s["metrics"]["rhyme_lines"] for s in evaluated)
    valid_stanzas = sum(s["metrics"]["valid_stanzas"] for s in evaluated)
    total_stanzas = sum(s["metrics"]["total_stanzas"] for s in evaluated)

    syllable_hist = Counter()
    for s in evaluated:
        syllable_hist.update(s["metrics"]["syllable_counts"])

    overlap_vals = [s["overlap"] for s in evaluated if s["overlap"] is not None]
    overlap = None
    if overlap_n is not None and overlap_vals:
        mean = sum(overlap_vals) / len(overlap_vals)
        # a near-duplicate generation has basically all n-grams in the corpus
        memorized = sum(1 for v in overlap_vals if v > 0.8)
        overlap = {
            "n": overlap_n,
            "mean": mean,
            "max": max(overlap_vals),
            "memorized_count": memorized,
        }

    return {
        "num_samples": len(samples),
        "evaluated": n_eval,
        "skipped_empty": skipped,
        "truncated": truncated,
        "per_sonnet": per_sonnet,
        "per_line": {
            "valid_hendecasyllables": {"count": valid_hend, "total": total_lines},
            "rhyme_lines": {"count": rhyme_lines, "total": total_lines},
            "valid_stanzas": {"count": valid_stanzas, "total": total_stanzas},
        },
        "syllable_histogram": dict(sorted(syllable_hist.items())),
        "overlap": overlap,
    }


def print_report(r, strict):
    """Render the aggregate report dict to stdout."""

    def line(label, count, total, ci=None):
        if not total:
            print(f"{label:22} {count}/{total} (n/a)")
            return
        s = f"{label:22} {count}/{total} ({count / total * 100:.1f}%)"
        if ci is not None:
            s += f"  95% CI [{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]"
        print(s)

    labels = {
        "is_14_lines": "14 lines:",
        "is_correct_structure": "Correct structure:",
        "is_valid_stanzas": "All stanzas valid:",
        "is_valid_rhyme_meter": "Valid rhyme+meter:",
        "is_valid_sonnet": "Valid SONNET:",
    }

    print("=" * 56)
    print(
        f"Samples evaluated: {r['evaluated']}/{r['num_samples']} "
        f"({r['skipped_empty']} empty, {r['truncated']} truncated)"
        + ("  [strict meter]" if strict else "")
    )
    for key, label in labels.items():
        m = r["per_sonnet"][key]
        line(label, m["count"], m["total"], m["ci95"])
    print("-" * 56)
    pl = r["per_line"]
    line(
        "Valid hendecasyll.:",
        pl["valid_hendecasyllables"]["count"],
        pl["valid_hendecasyllables"]["total"],
    )
    line("Rhyming lines:", pl["rhyme_lines"]["count"], pl["rhyme_lines"]["total"])
    line("Valid stanzas:", pl["valid_stanzas"]["count"], pl["valid_stanzas"]["total"])

    if r["syllable_histogram"]:
        print("-" * 56)
        print("Syllables/line histogram:")
        hist = r["syllable_histogram"]
        peak = max(hist.values())
        for count in sorted(hist):
            bar = "█" * max(1, round(40 * hist[count] / peak))
            mark = " *" if count == 11 else "  "
            print(f"  {count:2}{mark} {hist[count]:5} {bar}")

    if r["overlap"] is not None:
        ov = r["overlap"]
        print("-" * 56)
        print(
            f"Overlap ({ov['n']}-gram): mean {ov['mean'] * 100:.1f}% overlap, "
            f"max {ov['max'] * 100:.1f}%, "
            f"{ov['memorized_count']} near-duplicate (>80% overlap)"
        )
    print("=" * 56)
