import collections

from job_search.queries import style_for
from job_search.topical import POSTING_CHARS, check_params, generate_params, pick_specs

SPEC = {"spec_id": "s0004", "constraints": {"seniority": "Senior", "country": "Canada", "salary_min_usd": 140000}}
JOB = {"title": "Senior Backend Engineer", "text": "§" * (POSTING_CHARS + 500)}


def test_pick_specs_balances_styles_and_is_reproducible():
    specs = [{"spec_id": f"s{i:04d}", "constraints": {}} for i in range(1, 301)]
    picked = pick_specs(specs, 30)
    assert collections.Counter(style_for(s["spec_id"]) for s in picked) == {"keywords": 10, "sentence": 10,
                                                                             "casual": 10}
    assert picked == pick_specs(specs, 30)
    assert [s["spec_id"] for s in picked] == sorted(s["spec_id"] for s in picked)


def test_generate_params_trims_the_posting_and_states_constraints():
    prompt = generate_params(SPEC, JOB)["messages"][0]["content"]
    assert "Posting title: Senior Backend Engineer" in prompt
    assert "minimum yearly salary (USD): $140,000" in prompt
    assert prompt.count("§") == POSTING_CHARS


def test_check_params_sends_constraints_and_query():
    prompt = check_params(SPEC, "sr backend python canada $140k+")["messages"][0]["content"]
    assert "open to candidates in: Canada" in prompt
    assert prompt.endswith("Query: sr backend python canada $140k+")
