"""Turn a search query into structured filters with an LLM.

Claude Haiku 5.5 reads the query and returns the values it asks for, chosen
from the snapshot's own field values: seniority levels, job categories,
countries, employment types and a minimum yearly salary in USD. Lists may be
empty (not asked for) or hold several values ("junior or mid-level").

The prompt is written from the field definitions alone. It deliberately does
not reuse the synonym list the evaluation queries were generated with, which
would hand the system the answer key's phrasing.

Parses are cached in `data/parses/` (committed), one line per query with
timing and token use; an interrupted run continues where it stopped.

    uv run python -m job_search.systems.parse data/normalised/himalayas-2026-10-13.jsonl.gz
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

from job_search.normalise import snapshot_date
from job_search.specs import QUERY_LEVEL
from job_search.systems import add_arguments, load_jobs, load_queries, run_name

MODEL = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT = 0.10, 0.50  # USD per million tokens
MAX_TOKENS = 4000
# Each request is ~3,350 input tokens (the country list); 4 workers stay under
# the account's 500k input tokens per minute.
WORKERS = 4
MAX_RETRIES = 8
TIMEOUT = 60  # seconds; a parse takes ~1 s, so a longer wait is a hung connection
PARSES_DIR = Path("data/parses")
LEVELS = sorted(set(QUERY_LEVEL.values()), key=list(QUERY_LEVEL.values()).index)
FIELDS = ["seniority", "category", "country", "employment_type", "salary_min_usd"]

SYSTEM = """\
You turn job search queries into filters for a remote job board. Return only
what the query asks for; leave a field empty when the query does not mention
it. Never guess a value the query does not state or clearly imply.

Fields:
- seniority: the level(s) of the role. Leadership covers director, head of,
  VP and executive roles. Manager means a people-management role.
- categories: the broad job family or families the query is about.
- countries: where the searcher lives or wants the job to hire from.
- employment_types: the kind of contract.
- salary_min_usd: the minimum yearly pay in US dollars the query asks for,
  as an integer ("150k" is 150000); 0 when no minimum is given.

Every job on the board is remote, so "remote" is not a filter."""


def parses_path(normalised: Path, query_set: str) -> Path:
    return PARSES_DIR / f"{run_name(MODEL, query_set)}-{snapshot_date(normalised)}.jsonl"


def schema(categories: list[str], countries: list[str], employment_types: list[str]) -> dict:
    def enum_list(values):
        return {"type": "array", "items": {"type": "string", "enum": values}}

    return {
        "type": "object",
        "properties": {
            "seniority": enum_list(LEVELS),
            "categories": enum_list(categories),
            "countries": enum_list(countries),
            "employment_types": enum_list(employment_types),
            "salary_min_usd": {"type": "integer"},
        },
        "required": ["seniority", "categories", "countries", "employment_types", "salary_min_usd"],
        "additionalProperties": False,
    }


def field_values(jobs: list[dict]) -> tuple[list[str], list[str], list[str]]:
    categories = sorted({c for j in jobs for c in j["categories"]})
    countries = sorted({c for j in jobs for c in j["countries"]})
    employment_types = sorted({j["employment_type"] for j in jobs if j["employment_type"]})
    return categories, countries, employment_types


def request(query: str, output_schema: dict) -> dict:
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM,
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": output_schema}},
        "messages": [{"role": "user", "content": query}],
    }


def parse_one(client, query: str, output_schema: dict) -> dict:
    start = time.perf_counter()
    message = client.messages.create(**request(query, output_schema))
    seconds = time.perf_counter() - start
    if message.stop_reason != "end_turn":
        raise ValueError(f"stop_reason {message.stop_reason}")
    text = next(block.text for block in message.content if block.type == "text")
    return {
        "parsed": json.loads(text),
        "seconds": round(seconds, 2),
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
    }


def parse_all(client, queries: list[dict], output_schema: dict, out: Path, workers: int = WORKERS) -> None:
    done = set()
    if out.exists():
        done = {json.loads(line)["spec_id"] for line in out.read_text().splitlines() if line.strip()}
    todo = [q for q in queries if q["spec_id"] not in done]
    print(f"{len(done)} cached, {len(todo)} to parse", file=sys.stderr)
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def work(q):
        row = {"spec_id": q["spec_id"], **parse_one(client, q["query"], output_schema)}
        with lock, out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(workers) as pool:
        for n, _ in enumerate(pool.map(work, todo), 1):
            if n % 200 == 0:
                print(f"  {n}/{len(todo)}", file=sys.stderr)


# --- parsing accuracy -----------------------------------------------------


def as_constraints(parsed: dict) -> dict:
    """Parsed output in the spec's field names; lists stay lists."""
    return {
        "seniority": parsed["seniority"],
        "category": parsed["categories"],
        "country": parsed["countries"],
        "employment_type": parsed["employment_types"],
        "salary_min_usd": parsed["salary_min_usd"] or None,
    }


def field_outcome(expected, got) -> str:
    """correct, missed (asked for, not parsed), extra (parsed, not asked) or wrong."""
    if isinstance(got, list):
        got = got or None
        expected = [expected] if expected is not None else None
    if got == expected:
        return "correct"
    if expected is None:
        return "extra"
    return "missed" if got is None else "wrong"


def accuracy(queries: list[dict], parses: dict[str, dict]) -> dict:
    per_field = {f: collections.Counter() for f in FIELDS}
    all_correct = 0
    for q in queries:
        got = as_constraints(parses[q["spec_id"]]["parsed"])
        outcomes = {f: field_outcome(q["constraints"].get(f), got[f]) for f in FIELDS}
        for f, outcome in outcomes.items():
            per_field[f][outcome] += 1
        all_correct += all(o == "correct" for o in outcomes.values())
    n = len(queries)
    return {
        "queries": n,
        "all_fields_correct": round(all_correct / n, 4),
        "fields": {f: {"correct": round(c["correct"] / n, 4), **{k: v for k, v in c.items() if k != "correct"}}
                   for f, c in per_field.items()},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(parser)
    args = parser.parse_args(argv)

    jobs = load_jobs(args.normalised)
    queries = load_queries(args.normalised, args.queries)
    out = parses_path(args.normalised, args.queries)
    parse_all(anthropic.Anthropic(max_retries=MAX_RETRIES, timeout=TIMEOUT), queries, schema(*field_values(jobs)), out)

    rows = {r["spec_id"]: r for r in map(json.loads, out.read_text().splitlines())}
    tokens_in = sum(r["input_tokens"] for r in rows.values())
    tokens_out = sum(r["output_tokens"] for r in rows.values())
    seconds = sorted(r["seconds"] for r in rows.values())
    print(json.dumps({
        "parses": str(out),
        "cost_usd": round((tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1e6, 2),
        "latency_p50_p90_s": [seconds[len(seconds) // 2], seconds[9 * len(seconds) // 10]],
        **accuracy(queries, rows),
    }, indent=2))


if __name__ == "__main__":
    main()
