import json
from types import SimpleNamespace as NS

import pytest

from job_search.queries import (
    ambiguity,
    check_params,
    describe,
    freeze,
    generate,
    generate_params,
    groups,
    parse_output,
    run_batch,
    split_group,
    style_for,
)

SPEC = {"spec_id": "s0004", "constraints": {"seniority": "Senior", "country": "Canada", "salary_min_usd": 140000}}


def message(text, stop_reason="end_turn"):
    return NS(stop_reason=stop_reason, content=[NS(type="thinking"), NS(type="text", text=text)],
              usage=NS(input_tokens=100, output_tokens=20))


def test_describe_and_style():
    assert describe(SPEC["constraints"]) == (
        "- seniority: Senior\n- open to candidates in: Canada\n- minimum yearly salary (USD): $140,000"
    )
    assert [style_for(f"s000{i}") for i in range(1, 4)] == ["sentence", "casual", "keywords"]


def test_params_use_the_right_models_and_schemas():
    other = {"spec_id": "s0005", "constraints": {"category": "Sales", "employment_type": "Contractor"}}
    gen = generate_params([SPEC, other])
    assert gen["model"] == "claude-sonnet-5-5"
    assert gen["output_config"]["format"]["schema"]["required"] == ["queries"]
    assert gen["messages"][0]["content"] == (
        "1. style: sentence\n   - seniority: Senior\n   - open to candidates in: Canada\n"
        "   - minimum yearly salary (USD): $140,000\n\n"
        "2. style: casual\n   - job category: Sales\n   - employment type: Contractor"
    )

    chk = check_params(SPEC, "senior roles in Canada 140k+")
    assert chk["model"] == "claude-haiku-5-5"
    assert chk["messages"][0]["content"].endswith("Query: senior roles in Canada 140k+")


def test_parse_output():
    assert parse_output(message('{"query": "x"}')) == {"query": "x"}
    with pytest.raises(ValueError):
        parse_output(message('{"query": "x', stop_reason="max_tokens"))


class FakeBatches:
    def __init__(self, outputs):
        self.outputs = outputs
        self.created = []
        self.polls = 0

    def create(self, requests):
        self.created.append(requests)
        return NS(id="batch_1")

    def retrieve(self, batch_id):
        self.polls += 1
        status = "ended" if self.polls > 1 else "in_progress"
        return NS(processing_status=status, request_counts=NS(processing=1, succeeded=0))

    def results(self, batch_id):
        for cid, text in self.outputs.items():
            if text is None:
                yield NS(custom_id=cid, result=NS(type="errored"))
            else:
                yield NS(custom_id=cid, result=NS(type="succeeded", message=message(text)))


def fake_client(outputs):
    return NS(messages=NS(batches=FakeBatches(outputs)))


def test_groups_and_split_group():
    specs = [{"spec_id": f"s{i:04d}"} for i in range(1, 24)]
    grouped = groups(specs, size=10)
    assert list(grouped) == ["g0001", "g0002", "g0003"] and len(grouped["g0003"]) == 3

    group = specs[:3]
    output = {"queries": [{"item": 2, "query": " b "}, {"item": 1, "query": "a"}, {"item": 9, "query": "x"}]}
    assert split_group(group, output) == {"s0001": "a", "s0002": "b"}


def test_run_batch_resumes_without_resubmitting(tmp_path):
    client = fake_client({"s2": '{"query": "b"}', "s1": '{"query": "a"}', "s3": None})
    state = tmp_path / ".gen.batch.json"
    state.write_text(json.dumps({"batch_id": "batch_1"}))
    rows, usage = run_batch(client, {"s1": {}, "s2": {}, "s3": {}}, state, sleep=lambda _: None)

    assert client.messages.batches.created == []
    assert [r["custom_id"] for r in rows] == ["s1", "s2", "s3"]
    assert rows[0]["output"] == {"query": "a"} and rows[2]["status"] == "errored"
    assert usage == {"input_tokens": 200, "output_tokens": 40}


def test_generate_expands_groups_into_one_row_per_spec(tmp_path):
    specs_path = tmp_path / "specs-2026-10-13.jsonl"
    specs = [{"spec_id": f"s{i:04d}", "constraints": {"category": "Sales"}} for i in range(1, 13)]
    specs_path.write_text("".join(json.dumps(s) + "\n" for s in specs))
    answer = json.dumps({"queries": [{"item": n, "query": f"q{n}"} for n in range(1, 10)]})
    client = fake_client({"g0001": answer, "g0002": None})

    generate(client, specs_path, raw_dir=tmp_path, sleep=lambda _: None)

    rows = [json.loads(line) for line in (tmp_path / "generate-2026-10-13.jsonl").read_text().splitlines()]
    assert len(rows) == 12
    assert rows[0] == {"custom_id": "s0001", "group": "g0001", "status": "succeeded", "output": {"query": "q1"}}
    assert rows[9]["status"] == "missing"
    assert rows[10]["status"] == rows[11]["status"] == "errored"
    assert not (tmp_path / ".generate-2026-10-13.batch.json").exists()
    with pytest.raises(FileExistsError):
        generate(client, specs_path, raw_dir=tmp_path)


def test_freeze_keeps_only_matching_queries(tmp_path):
    specs_path = tmp_path / "specs-2026-10-13.jsonl"
    specs = [{"spec_id": f"s000{i}", "constraints": {"seniority": "Senior", "category": "Sales"}} for i in range(1, 5)]
    specs_path.write_text("".join(json.dumps(s) + "\n" for s in specs))
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "generate-2026-10-13.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [
        {"custom_id": "s0001", "status": "succeeded", "output": {"query": "senior sales"}},
        {"custom_id": "s0002", "status": "succeeded", "output": {"query": "sales jobs"}},
        {"custom_id": "s0003", "status": "errored"},
        {"custom_id": "s0004", "status": "succeeded", "output": {"query": "senior sales roles"}},
    ]))
    (raw / "check-2026-10-13.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [
        {"custom_id": "s0001", "status": "succeeded", "output": {"verdict": "match", "problem": ""}},
        {"custom_id": "s0002", "status": "succeeded", "output": {"verdict": "mismatch", "problem": "no level"}},
        {"custom_id": "s0004", "status": "errored"},
    ]))

    summary = freeze(specs_path, raw)

    kept = [json.loads(line) for line in (tmp_path / "queries-2026-10-13.jsonl").read_text().splitlines()]
    assert [(q["spec_id"], q["query"], q["style"]) for q in kept] == [("s0001", "senior sales", "sentence")]
    assert summary["dropped"] == {"mismatch": 1, "generation failed": 1, "check failed": 1}
    with pytest.raises(FileExistsError):
        freeze(specs_path, raw)



@pytest.mark.parametrize(
    ("constraints", "query", "reason"),
    [
        ({"country": "Canada", "salary_min_usd": 100000}, "jobs in Canada 100k+", "currency unclear"),
        ({"country": "Canada", "salary_min_usd": 100000}, "jobs in Canada $100k+", None),
        ({"country": "Canada", "salary_min_usd": 100000}, "jobs in Canada 100k+ USD", None),
        ({"country": "United States", "salary_min_usd": 100000}, "US jobs 100k+", None),
        ({"salary_min_usd": 100000}, "jobs 100k+", None),
        ({"country": "Georgia"}, "jobs in Georgia", "country is also a US state"),
        ({"country": "Canada"}, "jobs in Canada", None),
    ],
)
def test_ambiguity(constraints, query, reason):
    assert ambiguity(constraints, query) == reason
