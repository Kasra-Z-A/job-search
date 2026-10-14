import json
from types import SimpleNamespace as NS

import numpy as np

from job_search.systems.filtered import filter_outcome, rank_filtered
from job_search.systems.parse import accuracy, as_constraints, field_outcome, parse_all, schema


def job(id, **overrides):
    base = {"id": id, "seniority": ["Senior"], "categories": ["Developer"], "countries": ["Canada"],
            "worldwide": False, "employment_type": "Full Time", "salary_status": "stated",
            "salary_min_usd": 100_000, "salary_max_usd": 150_000}
    return {**base, **overrides}


PARSED = {"seniority": ["Senior"], "categories": [], "countries": ["Canada"],
          "employment_types": [], "salary_min_usd": 140_000}


def test_as_constraints_and_field_outcome():
    c = as_constraints({**PARSED, "salary_min_usd": 0})
    assert c["salary_min_usd"] is None and c["country"] == ["Canada"]
    assert field_outcome("Senior", ["Senior"]) == "correct"
    assert field_outcome(None, []) == "correct"
    assert field_outcome("Senior", []) == "missed"
    assert field_outcome(None, ["Senior"]) == "extra"
    assert field_outcome("Senior", ["Senior", "Manager"]) == "wrong"
    assert field_outcome(140_000, 150_000) == "wrong"
    assert field_outcome(140_000, None) == "missed"


def test_filter_outcome():
    filters = as_constraints(PARSED)
    assert filter_outcome(job("a"), filters) == "match"
    assert filter_outcome(job("b", countries=["Spain"]), filters) == "no"
    assert filter_outcome(job("c", salary_status="none", salary_max_usd=None), filters) == "unknown"
    # Several values in one field: any of them is enough.
    assert filter_outcome(job("d", seniority=["Mid-level"]), {"seniority": ["Senior", "Mid-level"]}) == "match"
    assert filter_outcome(job("e"), {}) == "match"


def test_rank_filtered_puts_known_matches_first_then_similarity():
    jobs = [
        job("unknown-close", salary_status="none", salary_max_usd=None),
        job("match-far"),
        job("match-close"),
        job("wrong-country", countries=["Spain"]),
    ]
    similarity = np.array([0.9, 0.1, 0.5, 1.0])
    assert rank_filtered(jobs, similarity, as_constraints(PARSED)) == ["match-close", "match-far", "unknown-close"]


def test_schema_restricts_values():
    s = schema(["Developer"], ["Canada"], ["Full Time"])
    assert s["properties"]["countries"]["items"]["enum"] == ["Canada"]
    assert s["properties"]["seniority"]["items"]["enum"] == ["Entry-level", "Mid-level", "Senior", "Manager", "Leadership"]


def test_accuracy():
    queries = [
        {"spec_id": "s1", "constraints": {"seniority": "Senior", "country": "Canada", "salary_min_usd": 140_000}},
        {"spec_id": "s2", "constraints": {"seniority": "Senior", "category": "Sales"}},
    ]
    parses = {"s1": {"parsed": PARSED}, "s2": {"parsed": PARSED}}
    a = accuracy(queries, parses)
    assert a["all_fields_correct"] == 0.5
    assert a["fields"]["country"] == {"correct": 0.5, "extra": 1}
    assert a["fields"]["category"] == {"correct": 0.5, "missed": 1}


def test_parse_all_resumes(tmp_path):
    out = tmp_path / "parses.jsonl"
    out.write_text(json.dumps({"spec_id": "s1", "parsed": PARSED}) + "\n")
    calls = []

    def create(**params):
        calls.append(params["messages"][0]["content"])
        return NS(stop_reason="end_turn", content=[NS(type="text", text=json.dumps(PARSED))],
                  usage=NS(input_tokens=10, output_tokens=5))

    client = NS(messages=NS(create=create))
    parse_all(client, [{"spec_id": "s1", "query": "a"}, {"spec_id": "s2", "query": "b"}], {}, out, workers=2)
    assert calls == ["b"]
    assert [json.loads(line)["spec_id"] for line in out.read_text().splitlines()] == ["s1", "s2"]
