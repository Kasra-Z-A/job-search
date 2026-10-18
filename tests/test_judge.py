import math

import pytest

from job_search.judge import collect, dcg, listing, pick_for_worksheet, score_query, weighted_kappa

JOB = {"title": "Backend Engineer", "company": "Acme", "seniority": ["Senior"], "categories": ["Developer"],
       "worldwide": False, "countries": ["Canada", "United States"], "employment_type": "Full Time",
       "salary_status": "stated", "salary_min_usd": 120000, "salary_max_usd": 150000}


def test_listing_shows_what_a_job_seeker_sees():
    text = listing(JOB)
    assert "Open to people in: Canada, United States" in text
    assert "Salary (USD): $120,000–$150,000 a year" in text
    unstated = listing({**JOB, "salary_status": "none", "employment_type": None, "worldwide": True})
    assert "Salary (USD): not stated" in unstated
    assert "Employment type: not stated" in unstated
    assert "Open to people in: worldwide" in unstated


def test_dcg_uses_gains_0_1_3():
    assert dcg([2, 1, 0]) == pytest.approx(3 + 1 / math.log2(3))


def test_score_query_against_the_best_pooled_ranking():
    judged = {"a": 2, "b": 1, "c": 0, "d": 2}
    perfect = score_query(["a", "d", "b", "c"], judged)
    assert perfect["ndcg@10"] == pytest.approx(1.0)
    assert perfect["good@10"] == pytest.approx(0.2)
    assert perfect["any_good@10"] == 1.0
    worst = score_query(["c", "x"], judged)  # an unjudged job counts as 0
    assert worst == {"ndcg@10": 0.0, "good@10": 0.0, "any_good@10": 0.0}
    assert score_query(["a"], {"a": 0, "b": 0}) is None


def test_worksheet_pick_balances_grades():
    grades = {(f"s{i}", f"j{i}"): i % 3 for i in range(30)}
    picked = pick_for_worksheet(grades, per_grade=4)
    assert len(picked) == 12
    assert sorted(grades[p] for p in picked) == [0] * 4 + [1] * 4 + [2] * 4
    assert picked == pick_for_worksheet(grades, per_grade=4)


def test_collect_reads_filled_grades_only():
    text = """## 001 · s0008

**Query:** entry level sales

<!-- https://example.com/a -->

Grade: 2

## 002 · s0019

<!-- https://example.com/b -->

Grade:

## 003 · s0020

<!-- https://example.com/c -->

Grade:0
"""
    assert collect(text) == {("s0008", "https://example.com/a"): 2, ("s0020", "https://example.com/c"): 0}


def test_weighted_kappa():
    assert weighted_kappa([0, 1, 2, 2], [0, 1, 2, 2]) == pytest.approx(1.0)
    assert weighted_kappa([0, 2, 0, 2], [2, 0, 2, 0]) < 0


def test_check_pairs():
    from job_search.judge import check_pairs
    grades = {(f"s{i % 5}", f"j{i}"): i % 3 for i in range(60)}
    rows = check_pairs(grades, [f"j{i}" for i in range(200)], per_grade=5, n_random=10)
    kinds = [r["kind"] for r in rows]
    assert kinds.count("second") == kinds.count("blind") == 15
    second = [(r["spec_id"], r["job_id"]) for r in rows if r["kind"] == "second"]
    assert second == [(r["spec_id"], r["job_id"]) for r in rows if r["kind"] == "blind"]
    pooled = {}
    for sid, jid in grades:
        pooled.setdefault(sid, set()).add(jid)
    assert all(r["job_id"] not in pooled[r["spec_id"]] for r in rows if r["kind"] == "random")


def test_blind_params_hide_listing_details():
    from job_search.judge import judge_params
    prompt = judge_params("backend jobs", JOB | {"text": "desc"}, blind=True)["messages"][0]["content"]
    assert "Salary" not in prompt and "Title: Backend Engineer" in prompt
    assert judge_params("q", JOB | {"text": "d"}, model="claude-sonnet-5-5")["model"] == "claude-sonnet-5-5"
