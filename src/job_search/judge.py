"""Relevance judgments: an LLM grades query-job pairs from the full posting.

The label-based test cannot judge topic, and the main system filters on the
very labels it checks. Here each system's top results for the topical
queries are pooled, and Claude Haiku 5.5 grades every pair 0, 1 or 2 against
the guideline in SYSTEM, reading the query, the listing details a job seeker
sees on Himalayas and the description. A hand-graded subset would measure
how far the judge can be trusted.

    uv run python -m job_search.judge pool data/normalised/himalayas-2026-10-13.jsonl.gz
    uv run python -m job_search.judge sample data/normalised/himalayas-2026-10-13.jsonl.gz -n 20
    uv run python -m job_search.judge run data/normalised/himalayas-2026-10-13.jsonl.gz
    uv run python -m job_search.judge score data/normalised/himalayas-2026-10-13.jsonl.gz
    uv run python -m job_search.judge worksheet data/normalised/himalayas-2026-10-13.jsonl.gz
    uv run python -m job_search.judge agree data/normalised/himalayas-2026-10-13.jsonl.gz
    uv run python -m job_search.judge checks data/normalised/himalayas-2026-10-13.jsonl.gz
    uv run python -m job_search.judge checks-report data/normalised/himalayas-2026-10-13.jsonl.gz

`run` grades the whole pool through the Message Batches API and resumes a
submitted batch when rerun. `score` gives each system's nDCG@10 with gains
0, 1 and 3 for grades 0, 1 and 2, the ideal ranking taken from every graded
job for the query, plus the share of grade-2 jobs in the top 10. `worksheet`
writes pairs to grade by hand, blind, the same number for each judge grade
so good matches are not rare; `agree` compares the two.

Without hand grades, `checks` tests the judge with other LLM runs in one
batch: `second` re-grades CHECK_PER_GRADE pairs per Haiku grade with Claude
Sonnet 5.5 (does a stronger model agree?), `blind` re-grades the same pairs
with Haiku but without the listing details (how much do Himalayas' labels
sway the grade?), and `random` pairs each query with jobs from outside its
pool (a judge that is not simply generous should give these 0).

The API key comes from ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

import anthropic

from job_search.normalise import snapshot_date
from job_search.evaluate import paired_bootstrap
from job_search.queries import MAX_TOKENS, _finish, load_jsonl, parse_output, run_batch, write_jsonl
from job_search.systems import load_jobs

MODEL = "claude-haiku-5-5"
SECOND_MODEL = "claude-sonnet-5-5"
CHECK_PER_GRADE = 100
RANDOM_PAIRS = 100
QUERY_SET = "topical"  # set per run with --set; every path below reads it at call time
SYSTEMS = ["bm25", "embeddings", "hybrid", "filtered", "textonly"]
DEPTH = 10
POSTING_CHARS = 6000
SEED = 20261017
JUDGE_DIR = Path("data/judgments")
GAIN = {0: 0, 1: 1, 2: 3}
WORKSHEET_PER_GRADE = 50

SYSTEM = """\
You grade search results for a remote job board. You get a job seeker's query
and one job posting. Decide how well the job matches the query.

The query has hard requirements (seniority, job category, where the person
lives, employment type, minimum yearly salary in US dollars, whichever it
states) and a topic: the kind of work the person wants.

Grades:
- 2, good match: the job meets every hard requirement the query states, and
  the work is what the query asks for.
- 1, partial match: the work is related but not quite the topic (frontend
  for a backend query, account management for a sales query), or a hard
  requirement is neither met nor contradicted because the posting does not
  say (no salary given).
- 0, no match: the job contradicts a hard requirement (a different level,
  not open to people where the person lives, a different employment type, a
  salary range entirely below the minimum), or the work is a different kind
  of job.

Rules:
- A salary requirement is met if the top of the posting's range reaches the
  minimum.
- "Open to people in X" is met if the job hires in X or worldwide.
- The listing details come from the job board and are sometimes wrong. If
  the description clearly says otherwise, trust the description.
- Judge only what the query asks. A requirement the query does not state
  never lowers the grade: a missing salary only matters if the query asks
  for a minimum salary.

First check each hard requirement, then the topic, then give the grade."""

SCHEMA = {
    "type": "object",
    "properties": {
        "requirements": {"type": "string"},
        "topic": {"type": "string"},
        "grade": {"type": "integer", "enum": [0, 1, 2]},
    },
    "required": ["requirements", "topic", "grade"],
    "additionalProperties": False,
}


def listing(job: dict) -> str:
    where = "worldwide" if job["worldwide"] else ", ".join(job["countries"]) or "not stated"
    if job["salary_status"] == "stated":
        salary = f"${job['salary_min_usd']:,}–${job['salary_max_usd']:,} a year"
    else:
        salary = "not stated"
    return "\n".join([
        f"Title: {job['title']}",
        f"Company: {job['company']}",
        f"Seniority: {', '.join(job['seniority']) or 'not stated'}",
        f"Category: {', '.join(job['categories']) or 'not stated'}",
        f"Open to people in: {where}",
        f"Employment type: {job['employment_type'] or 'not stated'}",
        f"Salary (USD): {salary}",
    ])


def judge_params(query: str, job: dict, model: str = MODEL, blind: bool = False) -> dict:
    details = (f"Title: {job['title']}\nCompany: {job['company']}\n(No other listing details: judge from the "
               f"description.)") if blind else listing(job)
    prompt = f"Query: {query}\n\nListing details:\n{details}\n\nDescription:\n{job['text'][:POSTING_CHARS]}"
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM,
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        "messages": [{"role": "user", "content": prompt}],
    }


def pool(normalised: Path, systems: list[str] = SYSTEMS, depth: int = DEPTH) -> list[dict]:
    """One row per (query, job) pair in any system's top `depth`, with the systems that returned it."""
    date = snapshot_date(normalised)
    found = collections.defaultdict(list)
    for system in systems:
        for run in load_jsonl(Path("data/runs") / f"{system}.{QUERY_SET}-{date}.jsonl"):
            for job_id in run["ranking"][:depth]:
                found[(run["spec_id"], job_id)].append(system)
    return [{"spec_id": sid, "job_id": jid, "systems": s} for (sid, jid), s in sorted(found.items())]


def pool_path(normalised: Path) -> Path:
    return JUDGE_DIR / f"pool.{QUERY_SET}-{snapshot_date(normalised)}.jsonl"


def load_queries(normalised: Path) -> dict[str, dict]:
    path = Path("data/eval") / f"{QUERY_SET}-{snapshot_date(normalised)}.jsonl"
    return {q["spec_id"]: q for q in load_jsonl(path)}


def grades_path(normalised: Path) -> tuple[Path, Path]:
    date = snapshot_date(normalised)
    return JUDGE_DIR / f"grades.{QUERY_SET}-{date}.jsonl", JUDGE_DIR / f".grades.{QUERY_SET}-{date}.batch.json"


def run(client, normalised: Path, **kw) -> None:
    results, state = grades_path(normalised)
    if results.exists():
        raise FileExistsError(f"{results} exists; judgments are never overwritten")
    jobs = {j["id"]: j for j in load_jobs(normalised)}
    queries = load_queries(normalised)
    pairs = load_jsonl(pool_path(normalised))
    reused = earlier_grades(normalised, queries)
    # Job ids are URLs, too long for a batch custom_id, so requests are numbered in pool order.
    requests = {f"p{i:05d}": judge_params(queries[p["spec_id"]]["query"], jobs[p["job_id"]])
                for i, p in enumerate(pairs) if (p["spec_id"], p["job_id"]) not in reused}
    print(f"{len(reused)} pairs graded before, {len(requests)} to grade", file=sys.stderr)
    rows, usage = run_batch(client, requests, state, **kw) if requests else ([], {})
    graded = [{**pairs[int(r["custom_id"][1:])], **r} for r in rows]
    graded += [{**p, **reused[(p["spec_id"], p["job_id"])]} for p in pairs if (p["spec_id"], p["job_id"]) in reused]
    _finish(sorted(graded, key=lambda r: (r["spec_id"], r["job_id"])), usage, results, state)


def earlier_grades(normalised: Path, queries: dict[str, dict]) -> dict[tuple[str, str], dict]:
    """Grades from the set this one extends, for pairs whose query text is unchanged."""
    from job_search.topical import SETS
    parent = SETS.get(QUERY_SET, {}).get("extends")
    if not parent:
        return {}
    date = snapshot_date(normalised)
    parent_queries = {q["spec_id"]: q["query"] for q in load_jsonl(Path("data/eval") / f"{parent}-{date}.jsonl")}
    return {(r["spec_id"], r["job_id"]): {"custom_id": f"{parent}:{r['custom_id']}", "status": r["status"],
                                          "output": r["output"]}
            for r in load_jsonl(JUDGE_DIR / f"grades.{parent}-{date}.jsonl")
            if r["status"] == "succeeded" and parent_queries.get(r["spec_id"]) == queries.get(r["spec_id"], {}).get("query")}


def load_grades(normalised: Path) -> dict[tuple[str, str], int]:
    return {(r["spec_id"], r["job_id"]): r["output"]["grade"]
            for r in load_jsonl(grades_path(normalised)[0]) if r["status"] == "succeeded"}


def dcg(grades: list[int]) -> float:
    return sum(GAIN[g] / math.log2(i + 2) for i, g in enumerate(grades))


def score_query(ranking: list[str], judged: dict[str, int], depth: int = DEPTH) -> dict | None:
    """nDCG@depth and share of good matches; None if no pooled job for the query has any relevance."""
    ideal = dcg(sorted(judged.values(), reverse=True)[:depth])
    if ideal == 0:
        return None
    top = [judged.get(job_id, 0) for job_id in ranking[:depth]]
    return {"ndcg@10": dcg(top) / ideal, "good@10": top.count(2) / depth, "any_good@10": float(2 in top)}


def score(normalised: Path, systems: list[str] = SYSTEMS) -> dict:
    date = snapshot_date(normalised)
    grades = load_grades(normalised)
    judged = collections.defaultdict(dict)
    for (sid, jid), g in grades.items():
        judged[sid][jid] = g
    per_system = {}
    for system in systems:
        runs = {r["spec_id"]: r["ranking"] for r in load_jsonl(Path("data/runs") / f"{system}.{QUERY_SET}-{date}.jsonl")}
        per_system[system] = {sid: score_query(runs[sid], judged[sid]) for sid in sorted(judged)}
    scored = [sid for sid in sorted(judged) if per_system[systems[0]][sid] is not None]
    summary = {system: {m: round(statistics.fmean(rows[sid][m] for sid in scored), 4)
                        for m in ("ndcg@10", "good@10", "any_good@10")} for system, rows in per_system.items()}
    nd = {s: [per_system[s][sid]["ndcg@10"] for sid in scored] for s in systems}
    return {
        "queries": len(judged),
        "scored": len(scored),
        "pairs_graded": len(grades),
        "grades": dict(sorted(collections.Counter(grades.values()).items())),
        "systems": summary,
        "filtered_vs_embeddings_ndcg": paired_bootstrap(nd["embeddings"], nd["filtered"]),
        "filtered_vs_textonly_ndcg": paired_bootstrap(nd["textonly"], nd["filtered"]),
    }


def worksheet_path(normalised: Path) -> Path:
    return JUDGE_DIR / f"hand-worksheet.{QUERY_SET}-{snapshot_date(normalised)}.md"


def pick_for_worksheet(grades: dict[tuple[str, str], int], per_grade: int = WORKSHEET_PER_GRADE,
                       seed: int = SEED) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    by_grade = collections.defaultdict(list)
    for pair, g in sorted(grades.items()):
        by_grade[g].append(pair)
    picked = [p for g in sorted(by_grade) for p in rng.sample(by_grade[g], min(per_grade, len(by_grade[g])))]
    rng.shuffle(picked)
    return picked


def worksheet(normalised: Path) -> str:
    jobs = {j["id"]: j for j in load_jobs(normalised)}
    queries = load_queries(normalised)
    head = [
        "# Grade search results by hand",
        "",
        "For each item, read the query and the job, then write a grade after `Grade:`.",
        f"The description is exactly what the judge read: the first {POSTING_CHARS:,} characters.",
        "",
        "- 2, good match: meets every requirement the query states, and the work is what it asks for.",
        "- 1, partial: related but not quite the topic, or a stated requirement the posting does not address",
        "  (no salary given when the query asks for one).",
        "- 0, no match: contradicts a stated requirement, or a different kind of job.",
        "",
        "Requirements the query does not state never lower the grade. All jobs are remote; salaries are",
        "yearly, in US dollars; a salary requirement is met if the top of the range reaches it.",
        "",
    ]
    items = []
    for n, (sid, jid) in enumerate(pick_for_worksheet(load_grades(normalised)), 1):
        job = jobs[jid]
        description = job["text"][:POSTING_CHARS] + ("\n[cut here]" if len(job["text"]) > POSTING_CHARS else "")
        items += [f"## {n:03d} · {sid}", "", f"**Query:** {queries[sid]['query']}", "",
                  listing(job), f"Link: {job['url']}", "", "Description:", "", description, "", f"<!-- {jid} -->", "",
                  "Grade: ", ""]
    return "\n".join(head + items)


ITEM = re.compile(r"^## \d+ · (s\d+)$.*?^<!-- (\S+) -->$.*?^Grade:[ \t]*([012])?[ \t]*$", re.M | re.S)


def collect(text: str) -> dict[tuple[str, str], int]:
    return {(sid, jid): int(g) for sid, jid, g in ITEM.findall(text) if g}


def weighted_kappa(a: list[int], b: list[int], k: int = 3) -> float:
    """Cohen's kappa with quadratic weights: 1 is perfect agreement, 0 is chance."""
    n = len(a)
    observed = sum((x - y) ** 2 for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    expected = sum(ca[i] * cb[j] * (i - j) ** 2 for i in range(k) for j in range(k)) / n ** 2
    return 1 - observed / expected if expected else 1.0


def agree(normalised: Path) -> dict:
    hand = collect(worksheet_path(normalised).read_text(encoding="utf-8"))
    grades = load_grades(normalised)
    pairs = sorted(hand)
    return compare_grades([grades[p] for p in pairs], [hand[p] for p in pairs])


def checks_path(normalised: Path) -> tuple[Path, Path]:
    date = snapshot_date(normalised)
    return JUDGE_DIR / f"checks.{QUERY_SET}-{date}.jsonl", JUDGE_DIR / f".checks.{QUERY_SET}-{date}.batch.json"


def check_pairs(grades: dict[tuple[str, str], int], job_ids: list[str], per_grade: int = CHECK_PER_GRADE,
                n_random: int = RANDOM_PAIRS, seed: int = SEED) -> list[dict]:
    """The pairs `checks` grades: a balanced sample twice (second judge, blind) and random pairs."""
    rng = random.Random(seed)
    by_grade = collections.defaultdict(list)
    for pair, g in sorted(grades.items()):
        by_grade[g].append(pair)
    sampled = [p for g in sorted(by_grade) for p in rng.sample(by_grade[g], min(per_grade, len(by_grade[g])))]
    pooled = collections.defaultdict(set)
    for sid, jid in grades:
        pooled[sid].add(jid)
    queries = sorted(pooled)
    rows = [{"kind": kind, "spec_id": sid, "job_id": jid} for kind in ("second", "blind") for sid, jid in sampled]
    for i in range(n_random):
        sid = queries[i % len(queries)]
        jid = rng.choice(job_ids)
        while jid in pooled[sid]:
            jid = rng.choice(job_ids)
        rows.append({"kind": "random", "spec_id": sid, "job_id": jid})
    return rows


def checks(client, normalised: Path, **kw) -> None:
    results, state = checks_path(normalised)
    if results.exists():
        raise FileExistsError(f"{results} exists; judgments are never overwritten")
    jobs = {j["id"]: j for j in load_jobs(normalised)}
    queries = load_queries(normalised)
    pairs = check_pairs(load_grades(normalised), sorted(jobs))
    options = {"second": {"model": SECOND_MODEL}, "blind": {"blind": True}, "random": {}}
    requests = {f"c{i:05d}": judge_params(queries[p["spec_id"]]["query"], jobs[p["job_id"]], **options[p["kind"]])
                for i, p in enumerate(pairs)}
    rows, usage = run_batch(client, requests, state, **kw)
    _finish([{**pairs[int(r["custom_id"][1:])], **r} for r in rows], usage, results, state)


def compare_grades(a: list[int], b: list[int]) -> dict:
    return {
        "pairs": len(a),
        "exact_agreement": round(sum(x == y for x, y in zip(a, b)) / len(a), 4),
        "within_one": round(sum(abs(x - y) <= 1 for x, y in zip(a, b)) / len(a), 4),
        "weighted_kappa": round(weighted_kappa(a, b), 4),
        "mean_difference": round(statistics.fmean(y - x for x, y in zip(a, b)), 4),
        "by_first_grade": {f"{g}": dict(sorted(collections.Counter(y for x, y in zip(a, b) if x == g).items()))
                           for g in range(3)},
    }


def checks_report(normalised: Path) -> dict:
    grades = load_grades(normalised)
    rows = [r for r in load_jsonl(checks_path(normalised)[0]) if r["status"] == "succeeded"]
    report = {}
    for kind in ("second", "blind"):
        pairs = [r for r in rows if r["kind"] == kind]
        report[f"haiku_vs_{kind}"] = compare_grades([grades[(r["spec_id"], r["job_id"])] for r in pairs],
                                                    [r["output"]["grade"] for r in pairs])
    random_grades = collections.Counter(r["output"]["grade"] for r in rows if r["kind"] == "random")
    report["random_pairs"] = {"pairs": sum(random_grades.values()), "grades": dict(sorted(random_grades.items()))}
    return report


def sample(client, normalised: Path, n: int) -> list[dict]:
    jobs = {j["id"]: j for j in load_jobs(normalised)}
    queries = load_queries(normalised)
    rows = []
    for pair in random.Random(SEED).sample(load_jsonl(pool_path(normalised)), n):
        query, job = queries[pair["spec_id"]]["query"], jobs[pair["job_id"]]
        out = parse_output(client.messages.create(**judge_params(query, job)))
        rows.append({**pair, "query": query, "title": job["title"], **out})
        print(f"[{out['grade']}] {query}\n    job: {job['title']} ({job['company']})\n"
              f"    requirements: {out['requirements']}\n    topic: {out['topic']}", file=sys.stderr)
    return rows


def main(argv: list[str] | None = None) -> None:
    global QUERY_SET
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["pool", "sample", "run", "score", "worksheet", "agree", "checks", "checks-report"])
    parser.add_argument("normalised", type=Path)
    parser.add_argument("-n", type=int, default=20)
    parser.add_argument("--set", dest="name", default=QUERY_SET, help="query set name in data/eval/")
    args = parser.parse_args(argv)
    QUERY_SET = args.name
    if args.command == "pool":
        rows = pool(args.normalised)
        write_jsonl(pool_path(args.normalised), rows)
        per_query = collections.Counter(r["spec_id"] for r in rows)
        print(json.dumps({"pool": str(pool_path(args.normalised)), "pairs": len(rows), "queries": len(per_query),
                          "pairs_per_query": round(len(rows) / len(per_query), 1)}, indent=2))
        return
    if args.command == "score":
        result = score(args.normalised)
        out = Path("data/results") / f"judged.{QUERY_SET}-{snapshot_date(args.normalised)}.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    if args.command == "worksheet":
        out = worksheet_path(args.normalised)
        if out.exists():
            raise FileExistsError(f"{out} exists; it may already hold hand grades")
        out.write_text(worksheet(args.normalised), encoding="utf-8")
        print(out)
        return
    if args.command == "agree":
        print(json.dumps(agree(args.normalised), indent=2))
        return
    if args.command == "checks-report":
        result = checks_report(args.normalised)
        out = Path("data/results") / f"judge-checks.{QUERY_SET}-{snapshot_date(args.normalised)}.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    if args.command == "checks":
        checks(anthropic.Anthropic(), args.normalised)
        return
    if args.command == "run":
        run(anthropic.Anthropic(), args.normalised)
        return
    rows = sample(anthropic.Anthropic(), args.normalised, args.n)
    out = JUDGE_DIR / f"sample.{QUERY_SET}-{snapshot_date(args.normalised)}.jsonl"
    write_jsonl(out, rows)
    print(out)


if __name__ == "__main__":
    main()
