from common import report
from common.eval import evaluate_structure


def _sample(index, text, truncated=False, empty=False, overlap=None):
    return {
        "index": index,
        "truncated": truncated,
        "empty": empty,
        "overlap": overlap,
        "text": text,
        "metrics": evaluate_structure(text),
    }


def test_aggregate_counts_empty_and_truncated():
    samples = [
        _sample(0, "casa\nmare", truncated=True),
        _sample(1, "", empty=True),
        _sample(2, "luna\nsole"),
    ]
    r = report.aggregate(samples, overlap_n=None)
    assert r["num_samples"] == 3
    assert r["evaluated"] == 2  # the empty one is excluded
    assert r["skipped_empty"] == 1
    assert r["truncated"] == 1
    assert r["per_sonnet"]["is_valid_sonnet"]["total"] == 2


def test_aggregate_reports_ci_and_histogram():
    samples = [_sample(i, "casa\nmare\nluna") for i in range(4)]
    r = report.aggregate(samples, overlap_n=None)
    ci = r["per_sonnet"]["is_14_lines"]["ci95"]
    assert 0.0 <= ci[0] <= ci[1] <= 1.0
    assert r["syllable_histogram"] == {2: 12}
    assert r["overlap"] is None


def test_aggregate_overlap_flags_memorized():
    samples = [
        _sample(0, "casa\nmare", overlap=0.95),  # near-duplicate
        _sample(1, "luna\nsole", overlap=0.10),
    ]
    r = report.aggregate(samples, overlap_n=4)
    assert r["overlap"]["memorized_count"] == 1
    assert abs(r["overlap"]["mean"] - 0.525) < 1e-9
    assert r["overlap"]["max"] == 0.95


def test_load_corpus_ngrams_strips_metadata(tmp_path):
    f = tmp_path / "s.txt"
    f.write_text(
        "<SONNET>\n<RHYME_A> ella | Te lo saressi creso\n<END>", encoding="utf-8"
    )
    grams = report.load_corpus_ngrams(str(tmp_path), 3)
    assert ("te", "lo", "saressi") in grams
    assert all("rhyme_a" not in g and "ella" not in g for g in grams)
