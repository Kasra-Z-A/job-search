import collections

import pytest

from job_search.human import collect, pick, worksheet


def spec(n, size):
    fields = ["seniority", "category", "country", "salary_min_usd"][:size]
    values = {"seniority": "Senior", "category": "Developer", "country": "Canada", "salary_min_usd": 140000}
    return {"spec_id": f"s{n:04d}", "constraints": {f: values[f] for f in fields}, "query": "llm", "style": "keywords"}


QUERIES = [spec(i, 2 + i % 3) for i in range(1, 101)]


def test_pick_keeps_the_constraint_mix_and_is_reproducible():
    picked = pick(QUERIES, n=30)
    sizes = collections.Counter(len(q["constraints"]) for q in picked)
    assert sizes == {2: 10, 3: 10, 4: 10}
    assert picked == pick(QUERIES, n=30)


def test_worksheet_round_trip():
    text = worksheet(QUERIES[:2])
    assert "## s0001\n\n- senior level\n- Developer jobs\n- open to people living in Canada\n\nQuery: " in text
    filled = text.replace("Query: \n", "Query: senior dev jobs canada\n", 1).replace("Query: \n", "Query: senior dev\n", 1)
    rows = collect(filled, QUERIES)
    assert [(r["spec_id"], r["query"], r["style"]) for r in rows] == [
        ("s0001", "senior dev jobs canada", "human"), ("s0002", "senior dev", "human")]
    assert rows[0]["constraints"] == QUERIES[0]["constraints"]


def test_collect_refuses_unfinished_worksheets():
    with pytest.raises(ValueError, match="no query written"):
        collect(worksheet(QUERIES[:2]), QUERIES)
