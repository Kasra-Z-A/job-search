"""Text-only track: recover a job's fields from its posting text with an LLM.

Claude Haiku 5.5 reads the title and description, never Himalayas' fields,
and returns what the text states: seniority, job category, where candidates
may live, employment type and salary with currency and pay period. Anything
the text does not state stays unknown instead of guessed. Salaries then go
through the same exchange rates and plausibility rules as the labels
(`normalise.normalise_salary`).

The output has the normalised job's field names, with None marking an
unknown seniority, location or employment type, so `filtered` can search it
unchanged. `accuracy` compares it with Himalayas' labels, which also shows
how much of each field the text carries at all.

    uv run python -m job_search.systems.extract sample data/normalised/himalayas-2026-10-13.jsonl.gz -n 20
    uv run python -m job_search.systems.extract run data/normalised/himalayas-2026-10-13.jsonl.gz
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
import sys
from pathlib import Path

import anthropic

from job_search.normalise import (
    SENIORITY_LEVELS,
    load_or_fetch_rates,
    normalise_salary,
    normalise_seniority,
    snapshot_date,
)
from job_search.queries import parse_output, run_batch
from job_search.systems import job_text, load_jobs
from job_search.systems.parse import field_values

MODEL = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT = 0.10, 0.50  # USD per million tokens, before the batch discount
MAX_TOKENS = 4000
MAX_CHARS = 20_000
EXTRACTED_DIR = Path("data/extracted")
PERIODS = ["annual", "monthly", "weekly", "hourly"]

SYSTEM = """\
You read a remote job posting and record facts the posting states or clearly
implies. When the posting does not say, leave the field empty or choose
"not stated". Do not guess from the company, the industry or what is typical.

Fields:
- seniority: the level(s) of the role. Manager means managing people;
  Director and Executive are director, head of, VP and C-level roles.
- categories: the broad job family or families of the role.
- location_rule: "worldwide" if candidates may live anywhere; "countries" if
  the posting limits where candidates may live (list every country allowed;
  for a region such as "EU" or "LATAM", list its countries); "not stated"
  if the posting does not say.
- employment_type: the kind of contract, or "not stated".
- salary_min, salary_max: the pay range as numbers in the posting's currency
  and period; 0 when no pay is given. Use the same number twice for a
  single amount.
- currency: ISO 4217 code of the pay, such as USD or EUR; empty if no pay.
- period: what the pay amount covers; empty if no pay."""


def schema(categories: list[str], countries: list[str], employment_types: list[str]) -> dict:
    def enum_list(values):
        return {"type": "array", "items": {"type": "string", "enum": values}}

    return {
        "type": "object",
        "properties": {
            "seniority": enum_list(SENIORITY_LEVELS),
            "categories": enum_list(categories),
            "location_rule": {"type": "string", "enum": ["worldwide", "countries", "not stated"]},
            "countries": enum_list(countries),
            "employment_type": {"type": "string", "enum": [*employment_types, "not stated"]},
            "salary_min": {"type": "number"},
            "salary_max": {"type": "number"},
            "currency": {"type": "string"},
            "period": {"type": "string", "enum": [*PERIODS, ""]},
        },
        "required": ["seniority", "categories", "location_rule", "countries", "employment_type",
                     "salary_min", "salary_max", "currency", "period"],
        "additionalProperties": False,
    }


def request(job: dict, output_schema: dict) -> dict:
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM,
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": output_schema}},
        "messages": [{"role": "user", "content": job_text(job)[:MAX_CHARS]}],
    }


def to_fields(job_id: str, out: dict, per_eur: dict[str, float]) -> dict:
    """Extraction output in the normalised job's shape; None means unknown."""
    if out["location_rule"] == "worldwide":
        countries, worldwide = [], True
    elif out["location_rule"] == "countries" and out["countries"]:
        countries, worldwide = sorted(set(out["countries"])), False
    else:
        countries, worldwide = None, False
    salary = normalise_salary({
        "minSalary": out["salary_min"] or None,
        "maxSalary": out["salary_max"] or None,
        "currency": out["currency"].upper() or None,
        "salaryPeriod": out["period"] or None,
    }, per_eur)
    return {
        "id": job_id,
        "seniority": normalise_seniority(out["seniority"]) or None,
        "categories": sorted(set(out["categories"])),
        "countries": countries,
        "worldwide": worldwide,
        "employment_type": None if out["employment_type"] == "not stated" else out["employment_type"],
        **salary,
    }


# --- accuracy against Himalayas' labels ----------------------------------


def compare_field(label, extracted) -> str:
    """correct, unknown (text silent) or wrong."""
    if extracted is None or extracted == []:
        return "unknown"
    return "correct" if extracted == label else "wrong"


def accuracy(labels: dict[str, dict], extracted: dict[str, dict]) -> dict:
    counts = {f: collections.Counter() for f in ["seniority", "category", "location", "employment_type", "salary"]}
    for job_id, x in extracted.items():
        j = labels[job_id]
        counts["seniority"][compare_field(j["seniority"], x["seniority"])] += 1
        if j["categories"]:
            hit = x["categories"] and set(x["categories"]) & set(j["categories"])
            counts["category"]["unknown" if not x["categories"] else "correct" if hit else "wrong"] += 1
        label_loc = "worldwide" if j["worldwide"] else tuple(j["countries"])
        x_loc = None if x["countries"] is None and not x["worldwide"] else (
            "worldwide" if x["worldwide"] else tuple(x["countries"]))
        counts["location"][compare_field(label_loc, x_loc)] += 1
        counts["employment_type"][compare_field(j["employment_type"], x["employment_type"])] += 1
        if j["salary_status"] == "stated":
            if x["salary_status"] != "stated":
                counts["salary"]["unknown"] += 1
            else:
                close = abs(x["salary_min_usd"] - j["salary_min_usd"]) <= 0.05 * j["salary_min_usd"]
                counts["salary"]["correct" if close else "wrong"] += 1
    return {f: {**dict(c), "n": sum(c.values())} for f, c in counts.items()}


# --- run ------------------------------------------------------------------


def paths(normalised: Path) -> tuple[Path, Path]:
    stem = f"{MODEL}-{snapshot_date(normalised)}"
    return EXTRACTED_DIR / f"{stem}.jsonl.gz", EXTRACTED_DIR / f".{stem}.batch.json"


def write_gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        for row in rows:
            gz.write((json.dumps(row, ensure_ascii=False) + "\n").encode())


def load_extracted(normalised: Path) -> dict[str, dict]:
    with gzip.open(paths(normalised)[0], "rt", encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["sample", "run"])
    parser.add_argument("normalised", type=Path)
    parser.add_argument("-n", type=int, default=20)
    args = parser.parse_args(argv)

    jobs = load_jobs(args.normalised)
    labels = {j["id"]: j for j in jobs}
    per_eur = load_or_fetch_rates(snapshot_date(args.normalised))
    output_schema = schema(*field_values(jobs))
    client = anthropic.Anthropic(max_retries=8)

    if args.command == "sample":
        extracted, tokens = {}, collections.Counter()
        for job in random.Random(1).sample(jobs, args.n):
            message = client.messages.create(**request(job, output_schema))
            tokens["in"] += message.usage.input_tokens
            tokens["out"] += message.usage.output_tokens
            extracted[job["id"]] = to_fields(job["id"], parse_output(message), per_eur)
        per_job_in, per_job_out = tokens["in"] / args.n, tokens["out"] / args.n
        full = len(jobs) * (per_job_in * PRICE_IN + per_job_out * PRICE_OUT) / 1e6 / 2
        print(json.dumps({"tokens_per_job": [round(per_job_in), round(per_job_out)],
                          "estimated_batch_cost_usd": round(full, 2),
                          "accuracy": accuracy(labels, extracted)}, indent=2))
        return

    out, state = paths(args.normalised)
    if out.exists():
        raise SystemExit(f"{out} exists; extractions are never overwritten")
    custom = {f"j{n:05d}": job for n, job in enumerate(jobs)}
    rows, usage = run_batch(client, {cid: request(job, output_schema) for cid, job in custom.items()}, state)
    extracted, failed = [], collections.Counter()
    for row in rows:
        job_id = custom[row["custom_id"]]["id"]
        if row["status"] == "succeeded":
            extracted.append({**to_fields(job_id, row["output"], per_eur), "raw": row["output"]})
        else:
            failed[row["status"]] += 1
            extracted.append({**to_fields(job_id, EMPTY, per_eur), "raw": None})
    write_gz(out, extracted)
    state.unlink(missing_ok=True)
    cost = (usage.get("input_tokens", 0) * PRICE_IN + usage.get("output_tokens", 0) * PRICE_OUT) / 1e6 / 2
    print(json.dumps({"extracted": str(out), "jobs": len(extracted), "failed": dict(failed),
                      "cost_usd": round(cost, 2), "accuracy": accuracy(labels, {r["id"]: r for r in extracted})},
                     indent=2))


# A failed extraction counts as a posting that states nothing.
EMPTY = {"seniority": [], "categories": [], "location_rule": "not stated", "countries": [],
         "employment_type": "not stated", "salary_min": 0, "salary_max": 0, "currency": "", "period": ""}


if __name__ == "__main__":
    main()
