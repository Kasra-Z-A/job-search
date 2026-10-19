import base64
import gzip
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from job_search.app import answer, followup
from job_search.app.bundle import excerpt, slim
from job_search.app.handler import DailyCounter, decode_history, handle
from job_search.app.search import Searcher, explain, understood_list


def job(id, **overrides):
    base = {"id": id, "title": f"Job {id}", "company": "Acme", "published_at": "2026-10-13T00:00:00+00:00",
            "expires_at": "2026-12-12T00:00:00+00:00",
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
    encoded = [f"Full text of {j['id']} – ünïcode".encode() for j in jobs]
    (tmp_path / "texts.bin").write_bytes(b"".join(encoded))
    np.save(tmp_path / "text_offsets.npy", np.cumsum([0, *map(len, encoded)], dtype=np.int64))
    (tmp_path / "values.json").write_text(json.dumps({
        "snapshot": "2026-10-13", "embedding_model": "m", "categories": ["Developer"],
        "countries": ["Canada", "Spain"], "employment_types": ["Full Time"]}))
    return tmp_path


def message(payload):
    return NS(stop_reason="end_turn", content=[NS(type="text", text=json.dumps(payload))],
              usage=NS(input_tokens=1, output_tokens=1))


class FakeAnthropic:
    """Answers each step by its system prompt and records what it was asked."""

    def __init__(self, answer_output=None, followup_output=None, fail_answer=False):
        self.answer_output = answer_output or {"summary": "Two strong fits at $150k.",
                                               "picks": [{"job": 2, "reason": "Pays up to $150k"},
                                                         {"job": 1, "reason": "Senior role in Canada"}]}
        self.followup_output = followup_output
        self.fail_answer = fail_answer
        self.calls = []
        self.messages = NS(create=self.create)

    def create(self, **params):
        system = params["system"]
        self.calls.append((system, params["messages"][0]["content"]))
        if system == answer.SYSTEM:
            if self.fail_answer:
                raise RuntimeError("overloaded")
            return message(self.answer_output)
        if system == followup.SYSTEM:
            return message(self.followup_output)
        return message(PARSED)


def fake_openai(vector):
    return NS(embeddings=NS(create=lambda model, input: NS(data=[NS(embedding=vector)])))


JOBS = [
    job("far"),
    job("close"),
    job("no-salary", salary_status="none", salary_min_usd=None, salary_max_usd=None),
    job("spain", countries=["Spain"]),
]
VECTORS = [[0.1, 1.0], [1.0, 0.0], [1.0, 0.1], [1.0, 0.0]]


def searcher(tmp_path, anthropic):
    return Searcher(write_bundle(tmp_path, JOBS, VECTORS), anthropic, fake_openai([1.0, 0.0]))


def test_excerpt_and_slim():
    assert excerpt("one two\nthree", limit=100) == "one two three"
    assert excerpt("word " * 100, limit=12) == "word word…"
    full = {**job("a"), "text": "Build things.", "subcategories": ["x"]}
    assert "subcategories" not in slim(full) and "text" not in slim(full)
    assert slim(full)["excerpt"] == "Build things."


def test_first_message_retrieves_answers_and_grounds(tmp_path):
    anthropic = FakeAnthropic()
    out = searcher(tmp_path, anthropic).search("senior jobs canada 140k+")

    # Parser first, then the answer, whose prompt carries the candidates' full text.
    assert anthropic.calls[1][0] == answer.SYSTEM
    assert "Full text of close – ünïcode" in anthropic.calls[1][1]
    assert out["counts"] == {"match": 2, "maybe": 1, "total": 4}
    assert [p["id"] for p in out["picks"]] == ["far", "close"]
    assert out["picks"][0]["reason"] == "Pays up to $150k"
    assert [m["id"] for m in out["more"]] == ["no-salary"]
    assert out["more"][0]["meets"] == {"seniority": "yes", "country": "yes", "salary_min_usd": "unknown"}
    assert out["grounding"] == [] and out["search_text"] == "senior jobs canada 140k+"
    assert "text" not in out["picks"][0]
    assert set(out["timing"]) == {"understand_ms", "retrieve_ms", "answer_ms"}


def test_followup_uses_history_and_can_require_salary(tmp_path):
    anthropic = FakeAnthropic(followup_output={**PARSED, "salary_min_usd": 0, "search_text": "senior roles in Canada",
                                               "require_salary": True})
    history = [{"q": "senior jobs canada", "filters": {**PARSED, "countries": ["Canada", "Mars"]},
                "picks": ["close", "unknown-id"]}]
    out = searcher(tmp_path, anthropic).search("only ones that list pay", history)

    system, prompt = anthropic.calls[0]
    assert system == followup.SYSTEM
    assert "1. Job close" in prompt and "Mars" not in prompt
    assert prompt.endswith("New message: only ones that list pay")
    assert out["require_salary"] and out["search_text"] == "senior roles in Canada"
    assert "no-salary" not in [j["id"] for j in out["picks"] + out["more"]]


def test_failed_answer_falls_back_to_the_ranking(tmp_path):
    out = searcher(tmp_path, FakeAnthropic(fail_answer=True)).search("senior jobs canada 140k+")
    assert out["summary"] == answer.FALLBACK_SUMMARY and out["grounding"] == ["answer failed"]
    assert [p["id"] for p in out["picks"]] == ["close", "far", "no-salary"]


def test_ground_removes_invented_amounts_and_bad_picks():
    candidates = [{"job": job("a", salary_min_usd=91_742, salary_max_usd=110_000)},
                  {"job": job("b", salary_status="none")}]
    output = {"summary": "One pays $250k.", "picks": [
        {"job": 1, "reason": "Pays $92k to $110k"}, {"job": 1, "reason": "again"},
        {"job": 7, "reason": "not a candidate"}, {"job": 2, "reason": "Around 300k, they say"}]}
    summary, picks, problems = answer.ground(output, candidates, salary_filter=90_000)
    assert summary == answer.FALLBACK_SUMMARY
    assert picks == [{"index": 0, "reason": "Pays $92k to $110k"}, {"index": 1, "reason": ""}]
    assert problems == ["summary amount", "repeated pick", "pick outside candidates", "reason amount"]


def test_amounts():
    assert answer.amounts("$92k, $110,000 and 150k+; 3 roles; $1.5k") == [92_000, 110_000, 150_000, 1_500]


def test_explain_and_understood_list():
    assert explain(job("a"), {"seniority": ["Senior"], "category": [], "salary_min_usd": None}) == {"seniority": "yes"}
    assert understood_list(PARSED, require_salary=True) == [
        "Senior", "open to Canada", "at least $140,000 a year", "salary stated"]


def test_clean_filters_keeps_only_allowed_values():
    enum = {"items": {"enum": ["Senior", "Canada", "Developer", "Full Time"]}}
    s = {"properties": {f: enum for f in ("seniority", "categories", "countries", "employment_types")}}
    cleaned = followup.clean_filters({"seniority": ["Senior", "Wizard"], "countries": "Canada",
                                      "salary_min_usd": -5}, s)
    assert cleaned == {"seniority": ["Senior"], "categories": [], "countries": [],
                       "employment_types": [], "salary_min_usd": 0}


class FakeSearcher:
    def __init__(self, fail=False):
        self.fail = fail
        self.history = "unset"

    def search(self, query, history=None):
        self.history = history
        if self.fail:
            raise RuntimeError("upstream down")
        return {"query": query}


class FakeCounter:
    def __init__(self, allow):
        self.allow = allow

    def take(self):
        return self.allow


def ctx(turns):
    return base64.urlsafe_b64encode(json.dumps(turns).encode()).decode().rstrip("=")


def test_handle_validates_caps_and_caches():
    ok = handle({"q": "  senior   dev  "}, FakeSearcher(), FakeCounter(True))
    assert ok["statusCode"] == 200 and json.loads(ok["body"]) == {"query": "senior dev"}
    assert "max-age" in ok["headers"]["cache-control"]

    assert handle({"q": "x"}, FakeSearcher(), FakeCounter(True))["statusCode"] == 400
    assert handle({}, FakeSearcher(), FakeCounter(True))["statusCode"] == 400
    capped = handle({"q": "senior dev"}, FakeSearcher(), FakeCounter(False))
    assert capped["statusCode"] == 429 and capped["headers"]["cache-control"] == "no-store"
    assert handle({"q": "senior dev"}, FakeSearcher(fail=True), None)["statusCode"] == 502


def test_handle_followup_needs_valid_context_and_is_not_cached():
    turn = {"q": "senior dev", "filters": PARSED, "picks": ["a", "b"]}
    fake = FakeSearcher()
    ok = handle({"q": "only full-time", "ctx": ctx([turn])}, fake, None, followup=True)
    assert ok["statusCode"] == 200 and ok["headers"]["cache-control"] == "no-store"
    assert fake.history == [turn]

    assert handle({"q": "only full-time"}, FakeSearcher(), None, followup=True)["statusCode"] == 400
    assert handle({"q": "only full-time", "ctx": "%%%"}, FakeSearcher(), None, followup=True)["statusCode"] == 400


def test_decode_history_rejects_bad_shapes():
    good = {"q": "a", "filters": {}, "picks": []}
    assert decode_history(ctx([good])) == [good]
    assert decode_history(ctx([good] * 5)) is None
    assert decode_history(ctx([{"q": 1, "filters": {}, "picks": []}])) is None
    assert decode_history(ctx({"q": "a"})) is None
    assert decode_history("x" * 7000) is None


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
