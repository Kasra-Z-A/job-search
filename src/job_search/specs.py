"""Query specs: field constraints whose answer sets are known by construction.

A spec is a set of constraints on normalised job fields, such as
{seniority: Senior, category: Developer, country: Germany}. Each spec is
sampled from a real job (its anchor), so it has at least one answer. Later, an
LLM rewrites each spec as a natural-language query; the spec stays the answer
key.

Matching a job against a spec gives one of three outcomes per constraint:

- yes: the job satisfies it.
- no: the job's field contradicts it.
- unknown: the job does not state the field (no salary, no category).

A job is **relevant** if every constraint says yes, and **ignored** if none
says no but at least one is unknown. Grading leaves ignored jobs out instead
of counting them as wrong. Answer sets are recomputed from the spec and the
normalised snapshot; each spec stores counts and a hash of its relevant ids so
any drift is caught.

Definitions:

- seniority: Director and Executive are merged into Leadership, because the
  spot-check found Executive tags unreliable. A job matches if any of its
  levels maps to the queried one.
- salary_min_usd: "at least X" matches a job whose stated range reaches X
  (salary_max_usd >= X).
- country: "open to people in X" matches jobs restricted to X and jobs open
  worldwide.
- category, employment_type: exact match on one value.

    uv run python -m job_search.specs data/normalised/himalayas-2026-10-13.jsonl.gz
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import random
from pathlib import Path

from job_search.normalise import snapshot_date

QUERY_LEVEL = {
    "Entry-level": "Entry-level",
    "Mid-level": "Mid-level",
    "Senior": "Senior",
    "Manager": "Manager",
    "Director": "Leadership",
    "Executive": "Leadership",
}
# Employment types too rare to query on their own.
EMPLOYMENT_TYPES = {"Full Time", "Contractor", "Part Time", "Intern"}
FIELDS = ["seniority", "category", "country", "employment_type", "salary_min_usd"]
# Single-field specs are left out: there are only a few dozen distinct ones,
# each with hundreds to thousands of answers, and a system that filters on the
# field gets them trivially right.
CONSTRAINT_COUNTS = {2: 0.4, 3: 0.4, 4: 0.2}
SALARY_SHARE = 0.4  # share of specs anchored on a job with a stated salary
SALARY_STEP = 10_000
# Thresholds come from the bottom of the anchor's range, kept to amounts
# people search for.
MIN_SALARY_THRESHOLD = 20_000
MAX_SALARY_THRESHOLD = 300_000
# Broad specs are kept for precision@10; recall is reported only up to 100.
MAX_RELEVANT = 300
SEED = 20261014
OUT_DIR = Path("data/eval")


# --- matching -------------------------------------------------------------


def check(job: dict, field: str, value) -> str:
    """'yes', 'no' or 'unknown' for one constraint."""
    if field == "seniority":
        return "yes" if value in {QUERY_LEVEL[s] for s in job["seniority"]} else "no"
    if field == "category":
        if not job["categories"]:
            return "unknown"
        return "yes" if value in job["categories"] else "no"
    if field == "country":
        return "yes" if job["worldwide"] or value in job["countries"] else "no"
    if field == "employment_type":
        if not job["employment_type"]:
            return "unknown"
        return "yes" if job["employment_type"] == value else "no"
    if field == "salary_min_usd":
        if job["salary_status"] != "stated":
            return "unknown"
        return "yes" if job["salary_max_usd"] >= value else "no"
    raise ValueError(f"unknown field {field!r}")


def judge(job: dict, constraints: dict) -> str:
    """'relevant', 'irrelevant' or 'ignored' for a whole spec."""
    outcomes = {check(job, field, value) for field, value in constraints.items()}
    if "no" in outcomes:
        return "irrelevant"
    return "ignored" if "unknown" in outcomes else "relevant"


def answer_sets(jobs: list[dict], constraints: dict) -> tuple[list[str], list[str]]:
    relevant, ignored = [], []
    for job in jobs:
        verdict = judge(job, constraints)
        if verdict == "relevant":
            relevant.append(job["id"])
        elif verdict == "ignored":
            ignored.append(job["id"])
    return relevant, ignored


def ids_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]


# --- sampling -------------------------------------------------------------


def candidate_constraints(job: dict, rng: random.Random) -> dict:
    """Every constraint this job could anchor, one value per field."""
    options = {"seniority": rng.choice([QUERY_LEVEL[s] for s in job["seniority"]])} if job["seniority"] else {}
    if job["categories"]:
        options["category"] = rng.choice(job["categories"])
    if job["countries"]:
        options["country"] = rng.choice(job["countries"])
    if job["employment_type"] in EMPLOYMENT_TYPES:
        options["employment_type"] = job["employment_type"]
    if job["salary_status"] == "stated":
        threshold = job["salary_min_usd"] // SALARY_STEP * SALARY_STEP
        if MIN_SALARY_THRESHOLD <= threshold <= MAX_SALARY_THRESHOLD:
            options["salary_min_usd"] = threshold
    return options


def spec_key(constraints: dict) -> str:
    return json.dumps(constraints, sort_keys=True)


def sample_specs(jobs: list[dict], n: int, seed: int = SEED, max_relevant: int = MAX_RELEVANT) -> list[dict]:
    rng = random.Random(seed)
    with_salary = [j for j in jobs if j["salary_status"] == "stated"]
    sizes, weights = zip(*CONSTRAINT_COUNTS.items())
    specs, seen = [], set()
    attempts = 0
    while len(specs) < n:
        attempts += 1
        if attempts > n * 50:
            raise RuntimeError(f"only {len(specs)} specs after {attempts} attempts")
        want_salary = rng.random() < SALARY_SHARE
        anchor = rng.choice(with_salary if want_salary else jobs)
        options = candidate_constraints(anchor, rng)
        size = rng.choices(sizes, weights)[0]
        if len(options) < size:
            continue
        fields = rng.sample(sorted(options), size)
        if want_salary and "salary_min_usd" in options and "salary_min_usd" not in fields:
            fields[-1] = "salary_min_usd"
        constraints = {f: options[f] for f in FIELDS if f in fields}
        key = spec_key(constraints)
        if key in seen:
            continue
        seen.add(key)

        relevant, ignored = answer_sets(jobs, constraints)
        if not 1 <= len(relevant) <= max_relevant:
            continue
        specs.append({
            "spec_id": f"s{len(specs) + 1:04d}",
            "constraints": constraints,
            "anchor": anchor["id"],
            "relevant": len(relevant),
            "ignored": len(ignored),
            "relevant_hash": ids_hash(relevant),
        })
    return specs


# --- run ------------------------------------------------------------------


def summarise(specs: list[dict]) -> dict:
    count = collections.Counter
    rel = sorted(s["relevant"] for s in specs)
    return {
        "specs": len(specs),
        "constraints_per_spec": dict(sorted(count(len(s["constraints"]) for s in specs).items())),
        "field_use": dict(count(f for s in specs for f in s["constraints"]).most_common()),
        "relevant_p10_p50_p90": [rel[len(rel) // 10], rel[len(rel) // 2], rel[9 * len(rel) // 10]],
        "with_ignored": sum(1 for s in specs if s["ignored"]),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("normalised", type=Path)
    parser.add_argument("-n", type=int, default=1500)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    with gzip.open(args.normalised, "rt", encoding="utf-8") as f:
        jobs = [json.loads(line) for line in f]
    specs = sample_specs(jobs, args.n)

    out = args.out_dir / f"specs-{snapshot_date(args.normalised)}.jsonl"
    if out.exists():
        raise SystemExit(f"{out} exists; frozen files are never regenerated")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(s) + "\n" for s in specs))
    print(json.dumps({"output": str(out), **summarise(specs)}, indent=2))


if __name__ == "__main__":
    main()
