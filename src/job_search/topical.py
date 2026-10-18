"""Topical queries: the frozen specs plus the kind of work a job seeker wants.

The main query set asks only for labels (seniority, category, country,
employment type, salary), so the main system, which filters on those labels,
is correct by construction. Here each query keeps its spec's constraints and
adds a topic taken from the spec's anchor posting: the kind of work and one
or two specifics (a skill, tool, domain or product type), general enough that
other postings can match too. Relevance to the topic is graded separately;
the constraints still give exact labels for parse accuracy.

Claude Sonnet 5.5 reads the anchor's title and the start of its description
and writes the query and a short `topic` phrase. Styles rotate as in
`queries`. Himalayas' labels are sometimes wrong (a marketing analyst filed
under Sales), and a query built on such a label contradicts itself, so the
model first says whether the constraints fit the posting; specs that do not
are skipped and counted.

Specs are drawn evenly across the three styles. Claude Haiku 5.5 then
checks each query against its constraints, as in `queries`, and also that it
adds one topic and nothing else. `freeze` keeps coherent, matching queries
and applies the same ambiguity rules as the main set.

    uv run python -m job_search.topical sample data/normalised/himalayas-2026-10-13.jsonl.gz -n 12
    uv run python -m job_search.topical generate data/normalised/himalayas-2026-10-13.jsonl.gz -n 96
    uv run python -m job_search.topical check data/normalised/himalayas-2026-10-13.jsonl.gz
    uv run python -m job_search.topical freeze data/normalised/himalayas-2026-10-13.jsonl.gz

`generate` and `check` run through the Message Batches API and resume a
submitted batch when rerun. The API key comes from ANTHROPIC_API_KEY.

Query sets are versioned (SETS). `topical-v2` adds queries from specs
`topical` did not use, written the same way, and its frozen file holds both,
each row naming its generator:

    uv run python -m job_search.topical generate data/normalised/himalayas-2026-10-13.jsonl.gz -n 300 --set topical-v2
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import anthropic

from job_search.normalise import snapshot_date
from job_search.queries import (CHECK_SCHEMA, CHECKER_MODEL, GENERATOR_MODEL, MAX_TOKENS, RAW_DIR, STYLES,
                                VOCABULARY, _finish, ambiguity, describe, load_jsonl, parse_output, run_batch,
                                style_for, write_jsonl)
from job_search.systems import load_jobs

SEED = 20261017
POSTING_CHARS = 4000
SETS = {
    "topical": {"generator": GENERATOR_MODEL, "extends": None},
    "topical-v2": {"generator": GENERATOR_MODEL, "extends": "topical"},
}

SYSTEM = f"""\
You write search queries for a remote job board, the way real job seekers type
them. You get a job posting, a set of constraints and a style. Write the query
a person who wants this kind of job might type.

The query must:
- express every constraint, with the same value;
- say what kind of work the person wants, with at most one specific taken
  from the posting: a skill, a tool or a domain, not several (for example
  "backend Python", "B2B SaaS onboarding", "paid social").

The topic must be broad enough that dozens of postings on a large job board
could match it: never name the company, its product or its team, and do not
copy the job title. Add no requirement beyond the constraints and the topic: no
benefits, company size, years of experience or extra countries. Every job on
the board is remote, so "remote" may appear but adds nothing.

{VOCABULARY}

Styles:
- keywords: a search-box string of at most 8 words, no sentence ("sr backend python canada 140k+").
- sentence: one complete request or question ("Show me...", "Which...", "Find...").
- casual: how someone types to a friend or a chatbot, first person, relaxed.
Real people rarely write "I am looking for".

First decide whether the constraints fit the posting: the job board's labels
are sometimes wrong, for example a marketing analyst filed under the Sales
category. If a constraint clearly contradicts what the job is, set coherent
to false and leave query and topic empty. Otherwise return the query and,
separately, the topic you added as a short phrase."""

CHECKER_SYSTEM = f"""\
You check search queries for a remote job board against the constraints they
were written from. A query must ask for exactly these constraints, every one
expressed with the same value, plus one topic: the kind of work the person
wants, with at most one specific such as a skill, tool or domain.

{VOCABULARY}

Mismatch:
- a constraint that is missing, or whose value changed: a different level,
  category, country or employment type, or a salary that is not the exact
  amount;
- no topic beyond the constraints;
- a topic that names a company, product or team, or that only one job could
  match;
- any other added requirement: benefits, company size, years of experience
  beyond what the seniority implies, another country.

Not a mismatch: the word "remote" (every job is remote), any wording from the
vocabulary for the same value, a topic that narrows the category, or casual
phrasing.

First write the problem (or "none"), then the verdict that follows from it."""

SCHEMA = {
    "type": "object",
    "properties": {"coherent": {"type": "boolean"}, "query": {"type": "string"}, "topic": {"type": "string"}},
    "required": ["coherent", "query", "topic"],
    "additionalProperties": False,
}


def generate_params(spec: dict, job: dict, model: str = GENERATOR_MODEL) -> dict:
    prompt = (f"Style: {style_for(spec['spec_id'])}\n\nConstraints:\n{describe(spec['constraints'])}\n\n"
              f"Posting title: {job['title']}\n\nPosting text:\n{job['text'][:POSTING_CHARS]}")
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM,
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        "messages": [{"role": "user", "content": prompt}],
    }


def check_params(spec: dict, query: str) -> dict:
    prompt = f"Constraints:\n{describe(spec['constraints'])}\n\nQuery: {query}"
    return {
        "model": CHECKER_MODEL,
        "max_tokens": MAX_TOKENS,
        "system": CHECKER_SYSTEM,
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": CHECK_SCHEMA}},
        "messages": [{"role": "user", "content": prompt}],
    }


def pick_specs(specs: list[dict], n: int, seed: int = SEED, exclude: frozenset = frozenset()) -> list[dict]:
    """n specs from the frozen set, the same number for each style, none of them in `exclude`."""
    rng = random.Random(seed)
    by_style = collections.defaultdict(list)
    for spec in specs:
        if spec["spec_id"] not in exclude:
            by_style[style_for(spec["spec_id"])].append(spec)
    picked = [s for style in STYLES for s in rng.sample(by_style[style], n // len(STYLES))]
    return sorted(picked, key=lambda s: s["spec_id"])


def frozen_specs(normalised: Path) -> list[dict]:
    return load_jsonl(Path("data/eval") / f"queries-{snapshot_date(normalised)}.jsonl")


def paths(normalised: Path, stage: str, name: str = "topical", raw_dir: Path = RAW_DIR) -> tuple[Path, Path]:
    date = snapshot_date(normalised)
    return raw_dir / f"{name}-{stage}-{date}.jsonl", raw_dir / f".{name}-{stage}-{date}.batch.json"


def used_specs(normalised: Path, name: str) -> frozenset:
    """Spec ids generated for the sets `name` extends, so a new version uses fresh specs."""
    used, parent = set(), SETS[name]["extends"]
    while parent:
        used |= {r["custom_id"] for r in load_jsonl(paths(normalised, "generate", parent)[0])}
        parent = SETS[parent]["extends"]
    return frozenset(used)


def generate(client, normalised: Path, n: int, name: str = "topical", **kw) -> None:
    results, state = paths(normalised, "generate", name)
    if results.exists():
        raise FileExistsError(f"{results} exists; raw outputs are never overwritten")
    jobs = {j["id"]: j for j in load_jobs(normalised)}
    model = SETS[name]["generator"]
    specs = pick_specs(frozen_specs(normalised), n, exclude=used_specs(normalised, name))
    requests = {s["spec_id"]: generate_params(s, jobs[s["anchor"]], model) for s in specs}
    rows, usage = run_batch(client, requests, state, **kw)
    _finish(rows, usage, results, state)


def check(client, normalised: Path, name: str = "topical", **kw) -> None:
    results, state = paths(normalised, "check", name)
    if results.exists():
        raise FileExistsError(f"{results} exists; raw outputs are never overwritten")
    specs = {s["spec_id"]: s for s in frozen_specs(normalised)}
    generated = [r for r in load_jsonl(paths(normalised, "generate", name)[0])
                 if r["status"] == "succeeded" and r["output"]["coherent"]]
    requests = {r["custom_id"]: check_params(specs[r["custom_id"]], r["output"]["query"]) for r in generated}
    rows, usage = run_batch(client, requests, state, **kw)
    _finish(rows, usage, results, state)


def freeze(normalised: Path, name: str = "topical") -> dict:
    specs = {s["spec_id"]: s for s in frozen_specs(normalised)}
    checked = {r["custom_id"]: r for r in load_jsonl(paths(normalised, "check", name)[0])}
    parent = SETS[name]["extends"]
    kept = []
    if parent:
        parent_generator = SETS[parent]["generator"]
        kept = [{"generator": parent_generator, **q}
                for q in load_jsonl(Path("data/eval") / f"{parent}-{snapshot_date(normalised)}.jsonl")]
    dropped = collections.Counter()
    for gen in load_jsonl(paths(normalised, "generate", name)[0]):
        sid = gen["custom_id"]
        spec, chk = specs[sid], checked.get(sid)
        if gen["status"] != "succeeded":
            dropped["generation failed"] += 1
        elif not gen["output"]["coherent"]:
            dropped["labels do not fit the posting"] += 1
        elif not chk or chk["status"] != "succeeded":
            dropped["check failed"] += 1
        elif chk["output"]["verdict"] != "match":
            dropped["mismatch"] += 1
        elif reason := ambiguity(spec["constraints"], gen["output"]["query"]):
            dropped[reason] += 1
        else:
            label_query = spec.pop("query")
            spec.pop("style")
            kept.append({"generator": SETS[name]["generator"], **spec, "query": gen["output"]["query"],
                         "topic": gen["output"]["topic"], "style": style_for(sid), "label_query": label_query})

    out = Path("data/eval") / f"{name}-{snapshot_date(normalised)}.jsonl"
    if out.exists():
        raise FileExistsError(f"{out} exists; the evaluation set is frozen")
    write_jsonl(out, kept)
    return {"output": str(out), "kept": len(kept), "dropped": dict(dropped),
            "generators": dict(collections.Counter(q.get("generator", SETS[name]["generator"]) for q in kept)),
            "styles": dict(collections.Counter(q["style"] for q in kept)),
            "constraints_per_query": dict(sorted(collections.Counter(len(q["constraints"]) for q in kept).items()))}


def sample(client, normalised: Path, n: int, name: str = "topical") -> list[dict]:
    jobs = {j["id"]: j for j in load_jobs(normalised)}
    rows = []
    for spec in pick_specs(frozen_specs(normalised), n, exclude=used_specs(normalised, name)):
        job = jobs[spec["anchor"]]
        out = parse_output(client.messages.create(**generate_params(spec, job, SETS[name]["generator"])))
        rows.append({"spec_id": spec["spec_id"], "style": style_for(spec["spec_id"]),
                     "constraints": spec["constraints"], "anchor": spec["anchor"],
                     "anchor_title": job["title"], "label_query": spec["query"], **out})
        if not out["coherent"]:
            print(f"{spec['spec_id']} skipped, labels do not fit: {describe(spec['constraints'])!r}"
                  f"   (anchor: {job['title']})", file=sys.stderr)
            continue
        print(f"{spec['spec_id']} [{rows[-1]['style']}] {out['query']}\n    topic: {out['topic']}"
              f"   (anchor: {job['title']})", file=sys.stderr)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["sample", "generate", "check", "freeze"])
    parser.add_argument("normalised", type=Path)
    parser.add_argument("-n", type=int, default=12, help="specs for `sample` and `generate`")
    parser.add_argument("--set", dest="name", default="topical", choices=sorted(SETS))
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(json.dumps(freeze(args.normalised, args.name), indent=2))
        return
    client = anthropic.Anthropic()
    if args.command == "generate":
        generate(client, args.normalised, args.n, args.name)
    elif args.command == "check":
        check(client, args.normalised, args.name)
    else:
        rows = sample(client, args.normalised, args.n, args.name)
        out = RAW_DIR / f"{args.name}-sample-{snapshot_date(args.normalised)}.jsonl"
        write_jsonl(out, rows)
        print(out)


if __name__ == "__main__":
    main()
