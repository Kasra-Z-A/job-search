import pytest

from job_search.normalise import SENIORITY_LEVELS
from job_search.spotcheck import render_markdown, sample_jobs, summarise, years_mentioned


def test_years_mentioned_finds_experience_phrases():
    text = "About us.\nYou have 5+ years of experience in Python. Benefits: 30 days off."
    assert years_mentioned(text) == "You have 5+ years of experience in Python"
    assert "3-5 years" in years_mentioned("Requires 3-5 years in sales")
    assert years_mentioned("No experience needed.") == ""


def test_sample_takes_each_level_without_reusing_jobs():
    jobs = [{"id": f"{level}-{i}", "seniority": [level], "title": "t", "company": "c", "text": "x", "url": "u"}
            for level in SENIORITY_LEVELS for i in range(10)]
    jobs.append({"id": "all", "seniority": SENIORITY_LEVELS, "title": "t", "company": "c", "text": "x", "url": "u"})

    rows = sample_jobs(jobs, per_level=3, seed=1)

    assert len(rows) == 18
    assert len({r["url"] + r["tag"] + str(r["n"]) for r in rows}) == 18
    assert [r["n"] for r in rows] == list(range(1, 19))
    assert rows == sample_jobs(jobs, per_level=3, seed=1)


def test_summary_counts_verdicts_and_corrections():
    rows = [
        {"n": "1", "tag": "Senior", "verdict": "ok", "should_be": ""},
        {"n": "2", "tag": "Senior", "verdict": "Wrong", "should_be": "Mid-level"},
        {"n": "3", "tag": "Entry-level", "verdict": "unclear", "should_be": ""},
        {"n": "4", "tag": "Entry-level", "verdict": "", "should_be": ""},
    ]
    s = summarise(rows)
    assert s["judged"] == 3 and s["pending"] == 1
    assert s["agreement"] == 0.5
    assert s["corrections"] == [("Senior -> Mid-level", 1)]

    with pytest.raises(ValueError):
        summarise([{"n": "1", "tag": "Senior", "verdict": "maybe", "should_be": ""}])


def test_render_puts_wrong_and_unclear_first():
    base = {"title": "T", "company": "C", "url": "u", "years_mentioned": "", "text_start": "a / b", "note": ""}
    rows = [
        {**base, "n": "1", "tag": "Senior", "verdict": "ok", "should_be": ""},
        {**base, "n": "2", "tag": "Senior", "verdict": "unclear", "should_be": ""},
        {**base, "n": "3", "tag": "Executive", "verdict": "wrong", "should_be": "Senior", "note": "a recruiter"},
    ]
    md = render_markdown(rows, "Check")
    assert md.index("## #3 · wrong → should be Senior") < md.index("## #2 · unclear") < md.index("## #1 · ok")
    assert "> a recruiter" in md
    assert "a\n\nb" in md
