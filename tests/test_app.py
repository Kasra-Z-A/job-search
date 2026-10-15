import gzip
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from job_search.app.bundle import excerpt, slim
from job_search.app.handler import DailyCounter, handle
from job_search.app.search import Searcher, explain


def job(id, **overrides):
    base = {"id": id, "title": f"Job {id}", "company": "Acme", "published_at": "2026-10-13T00:00:00+00:00",
            "seniority": ["Senior"], "categories": ["Developer"], "countries": ["Canada"], "worldwide": False,
            "employment_type": "Full Time", "salary_status": "stated", "salary_min_usd": 100_000,
            "salary_max_usd": 150_000, "excerpt": "..."}
    return {**base, **overrides}


PARSED = {"seniority": ["Senior"], "categories": [], "countries": ["Canada"],
          "employment_types": [], "salary_min_usd": 140_000}


def write_bundle(tmp_path, jobs, vectors):
    with gzip.open(tmp_path / "jobs.json.gz", "wt") as f:
        json.dump(jobs, f)
    np.save(tmp_path / "vectors.npy", np.asarray(vectors, dtype=np.float16))
    (tmp_path / "values.json").write_text(json.dumps({
        "snapshot": "2026-10-13", "embedding_model": "m", "categories": ["Developer"],
        "countries": ["Canada", "Spain"], "employment_types": ["Full Time"]}))
    return tmp_path


def fake_clients(query_vector):
    def create(**params):
        return NS(stop_reason="end_turn", content=[NS(type="text", text=json.dumps(PARSED))],
                  usage=NS(input_tokens=1, output_tokens=1))

    anthropic = NS(messages=NS(create=create))
    openai = NS(embeddings=NS(create=lambda model, input: NS(data=[NS(embedding=query_vector)])))
    return anthropic, openai


def test_excerpt_and_slim():
    assert excerpt("one two\nthree", limit=100) == "one two three"
    assert excerpt("word " * 100, limit=12) == "word word…"
    full = {**job("a"), "text": "Build things.", "subcategories": ["x"]}
    assert "subcategories" not in slim(full) and slim(full)["excerpt"] == "Build things."


def test_search_filters_ranks_and_explains(tmp_path):
    jobs = [
        job("far"),
        job("close"),
        job("no-salary", salary_status="none", salary_min_usd=None, salary_max_usd=None),
        job("spain", countries=["Spain"]),
    ]
    bundle = write_bundle(tmp_path, jobs, [[0.1, 1.0], [1.0, 0.0], [1.0, 0.1], [1.0, 0.0]])
    searcher = Searcher(bundle, *fake_clients([1.0, 0.0]))

    out = searcher.search("senior jobs canada 140k+")

    assert [r["id"] for r in out["results"]] == ["close", "far", "no-salary"]
    assert out["counts"] == {"match": 2, "maybe": 1, "total": 4}
    assert out["results"][2]["meets"] == {"seniority": "yes", "country": "yes", "salary_min_usd": "unknown"}
    assert set(out["timing"]) == {"understand_ms", "embed_ms", "rank_ms"}


def test_explain_skips_unrequested_fields():
    assert explain(job("a"), {"seniority": ["Senior"], "category": [], "salary_min_usd": None}) == {"seniority": "yes"}


class FakeSearcher:
    def __init__(self, fail=False):
        self.fail = fail

    def search(self, query):
        if self.fail:
            raise RuntimeError("upstream down")
        return {"query": query}


class FakeCounter:
    def __init__(self, allow):
        self.allow = allow

    def take(self):
        return self.allow


def test_handle_validates_caps_and_caches():
    ok = handle({"q": "  senior   dev  "}, FakeSearcher(), FakeCounter(True))
    assert ok["statusCode"] == 200 and json.loads(ok["body"]) == {"query": "senior dev"}
    assert "max-age" in ok["headers"]["cache-control"]

    assert handle({"q": "x"}, FakeSearcher(), FakeCounter(True))["statusCode"] == 400
    assert handle({}, FakeSearcher(), FakeCounter(True))["statusCode"] == 400
    capped = handle({"q": "senior dev"}, FakeSearcher(), FakeCounter(False))
    assert capped["statusCode"] == 429 and capped["headers"]["cache-control"] == "no-store"
    assert handle({"q": "senior dev"}, FakeSearcher(fail=True), None)["statusCode"] == 502


def test_daily_counter_stops_at_the_limit():
    from botocore.exceptions import ClientError

    class Table:
        def __init__(self):
            self.count = 0

        def update_item(self, **kw):
            if self.count >= kw["ExpressionAttributeValues"][":limit"]:
                raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "UpdateItem")
            self.count += 1

    counter = DailyCounter(Table(), limit=2)
    assert [counter.take("2026-10-15") for _ in range(3)] == [True, True, False]


def test_daily_counter_raises_other_errors():
    from botocore.exceptions import ClientError

    class Broken:
        def update_item(self, **kw):
            raise ClientError({"Error": {"Code": "AccessDeniedException"}}, "UpdateItem")

    with pytest.raises(ClientError):
        DailyCounter(Broken(), 5).take("2026-10-15")
