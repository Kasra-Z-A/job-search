import math

import pytest

from job_search.evaluate import (
    compare,
    condense,
    load_run,
    paired_bootstrap,
    reference_runs,
    score_query,
    score_run,
    summarise,
)


def test_condense_drops_ignored_and_repeats():
    assert condense(["a", "x", "b", "a", "c"], ignored={"x"}) == ["a", "b", "c"]


def test_perfect_ranking_scores_one_even_with_few_relevant():
    s = score_query(["a", "b", "c", "z"], relevant={"a", "b", "c"}, ignored=set())
    assert s == {"precision@10": 1.0, "ndcg@10": 1.0, "recall@100": 1.0}


def test_ignored_jobs_count_neither_way():
    # Ignored jobs ahead of the correct ones are removed before scoring.
    s = score_query(["i1", "i2", "a", "b"], relevant={"a", "b"}, ignored={"i1", "i2"})
    assert s["precision@10"] == 1.0 and s["ndcg@10"] == 1.0


def test_wrong_jobs_lower_precision_and_ndcg():
    relevant = {f"r{i}" for i in range(20)}
    ranking = ["w0", "r0", "w1", "r1"] + [f"w{i}" for i in range(2, 10)]
    s = score_query(ranking, relevant, ignored=set())
    assert s["precision@10"] == 0.2
    ideal = sum(1 / math.log2(i + 2) for i in range(10))
    assert s["ndcg@10"] == pytest.approx((1 / math.log2(3) + 1 / math.log2(5)) / ideal)
    assert s["recall@100"] == 0.1


def test_recall_only_for_small_answer_sets():
    relevant = {f"r{i}" for i in range(101)}
    assert score_query(["r0"], relevant, set())["recall@100"] is None


def test_load_run_rejects_incomplete_or_foreign_runs(tmp_path):
    queries = [{"spec_id": "s1"}, {"spec_id": "s2"}]
    path = tmp_path / "run-2026-10-13.jsonl"

    path.write_text('{"spec_id": "s1", "ranking": ["a"]}\n')
    with pytest.raises(ValueError, match="no ranking"):
        load_run(path, queries, corpus={"a"})

    path.write_text('{"spec_id": "s1", "ranking": ["a"]}\n{"spec_id": "s2", "ranking": ["nope"]}\n')
    with pytest.raises(ValueError, match="not in the snapshot"):
        load_run(path, queries, corpus={"a"})


QUERIES = [
    {"spec_id": "s1", "style": "keywords", "constraints": {"seniority": "Senior", "category": "Sales"},
     "relevant_ids": {"a", "b"}, "ignored_ids": {"i"}},
    {"spec_id": "s2", "style": "casual", "constraints": {"seniority": "Senior", "country": "Spain", "category": "Design"},
     "relevant_ids": {"c"}, "ignored_ids": set()},
]


def test_reference_runs_bracket_the_scale():
    runs = reference_runs(QUERIES, corpus={"a", "b", "c", "i", "w"})
    oracle = summarise(score_run({r["spec_id"]: r["ranking"] for r in runs["oracle"]}, QUERIES))
    assert oracle["overall"] == {"queries": 2, "precision@10": 1.0, "ndcg@10": 1.0, "recall@100": 1.0}


def test_summary_slices():
    run = {"s1": ["a", "w"], "s2": ["w"]}
    summary = summarise(score_run(run, QUERIES))
    assert summary["slices"]["style=keywords"]["precision@10"] == 0.5
    assert summary["slices"]["style=casual"]["precision@10"] == 0.0
    assert summary["slices"]["field=seniority"]["queries"] == 2
    assert summary["slices"]["constraints=3"]["queries"] == 1


def test_paired_bootstrap():
    clear = paired_bootstrap([0.1] * 50, [0.6] * 50, samples=500)
    assert clear["difference"] == 0.5 and clear["ci95"] == [0.5, 0.5] and clear["result"]

    noise = paired_bootstrap([0.0, 1.0] * 25, [1.0, 0.0] * 25, samples=500)
    assert noise["difference"] == 0.0 and noise["ci95"][0] < 0 < noise["ci95"][1] and not noise["result"]


def test_compare_skips_queries_without_recall():
    a = [{"spec_id": "s1", "precision@10": 0.0, "ndcg@10": 0.0, "recall@100": None}]
    b = [{"spec_id": "s1", "precision@10": 1.0, "ndcg@10": 1.0, "recall@100": None}]
    out = compare(a, b)
    assert out["precision@10"]["difference"] == 1.0
    assert out["recall@100"]["queries"] == 0


def test_default_paths_follow_the_query_set_in_the_run_name():
    from pathlib import Path

    from job_search.evaluate import default_paths

    assert default_paths(Path("data/runs/bm25-2026-10-13.jsonl"))[0].name == "queries-2026-10-13.jsonl"
    assert default_paths(Path("data/runs/filtered.sloppy-2026-10-13.jsonl"))[0].name == "sloppy-2026-10-13.jsonl"
