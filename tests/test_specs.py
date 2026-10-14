import random

import pytest

from job_search.specs import answer_sets, candidate_constraints, check, judge, sample_specs


def job(id="j", **overrides):
    base = {
        "id": id,
        "seniority": ["Senior"],
        "categories": ["Developer"],
        "countries": ["Germany"],
        "worldwide": False,
        "employment_type": "Full Time",
        "salary_status": "stated",
        "salary_min_usd": 80_000,
        "salary_max_usd": 100_000,
    }
    return {**base, **overrides}


@pytest.mark.parametrize(
    ("overrides", "field", "value", "expected"),
    [
        ({}, "seniority", "Senior", "yes"),
        ({"seniority": ["Executive"]}, "seniority", "Leadership", "yes"),
        ({"seniority": ["Director"]}, "seniority", "Senior", "no"),
        ({}, "category", "Sales", "no"),
        ({"categories": []}, "category", "Sales", "unknown"),
        ({}, "country", "Germany", "yes"),
        ({}, "country", "Spain", "no"),
        ({"countries": [], "worldwide": True}, "country", "Spain", "yes"),
        ({}, "employment_type", "Contractor", "no"),
        ({}, "salary_min_usd", 100_000, "yes"),
        ({}, "salary_min_usd", 110_000, "no"),
        ({"salary_status": "none", "salary_max_usd": None}, "salary_min_usd", 50_000, "unknown"),
        ({"salary_status": "implausible", "salary_max_usd": None}, "salary_min_usd", 50_000, "unknown"),
    ],
)
def test_check(overrides, field, value, expected):
    assert check(job(**overrides), field, value) == expected


def test_judge_no_beats_unknown():
    no_salary = job(salary_status="none", salary_max_usd=None)
    assert judge(no_salary, {"category": "Developer", "salary_min_usd": 50_000}) == "ignored"
    assert judge(no_salary, {"category": "Sales", "salary_min_usd": 50_000}) == "irrelevant"
    assert judge(job(), {"category": "Developer", "salary_min_usd": 50_000}) == "relevant"


def test_answer_sets_split_relevant_and_ignored():
    jobs = [job("a"), job("b", salary_status="none", salary_max_usd=None), job("c", seniority=["Mid-level"])]
    assert answer_sets(jobs, {"seniority": "Senior", "salary_min_usd": 90_000}) == (["a"], ["b"])


def test_candidate_salary_threshold_rounds_down():
    options = candidate_constraints(job(salary_min_usd=127_500, salary_max_usd=150_000), random.Random(0))
    assert options["salary_min_usd"] == 120_000
    for low, high in [(15_000, 30_000), (320_000, 400_000)]:
        options = candidate_constraints(job(salary_min_usd=low, salary_max_usd=high), random.Random(0))
        assert "salary_min_usd" not in options
    assert "employment_type" not in candidate_constraints(job(employment_type="Volunteer"), random.Random(0))


def test_sample_specs_are_unique_answerable_and_reproducible():
    rng = random.Random(1)
    jobs = [
        job(f"j{i}",
            seniority=[rng.choice(["Entry-level", "Mid-level", "Senior", "Manager", "Director"])],
            categories=[rng.choice(["Developer", "Sales", "Design"])],
            countries=[rng.choice(["Germany", "Spain", "United States"])],
            salary_min_usd=rng.randrange(40_000, 150_000), salary_max_usd=200_000)
        for i in range(300)
    ]
    specs = sample_specs(jobs, 40, seed=7, max_relevant=50)

    assert len({str(sorted(s["constraints"].items())) for s in specs}) == 40
    for s in specs:
        relevant, _ = answer_sets(jobs, s["constraints"])
        assert 1 <= len(relevant) == s["relevant"] <= 50
        assert s["anchor"] in relevant
    assert specs == sample_specs(jobs, 40, seed=7, max_relevant=50)
