"""Score search runs against the frozen evaluation queries.

A run is a JSONL file with one line per query, written by a search system:

    {"spec_id": "s0001", "ranking": ["<job id>", "<job id>", ...]}

Rankings are over the normalised snapshot's job ids, best first, up to DEPTH
long. Answer sets are recomputed from each query's constraints and checked
against the counts and hash frozen with the query, so a changed snapshot or
matching rule is caught instead of silently changing scores.

Scoring per query:

- Ignored jobs (an unknown field, see `specs`) are removed from the ranking
  before scoring, so they count neither for nor against a system.
- precision@10: correct jobs in the top 10, divided by min(10, relevant), so
  a perfect ranking scores 1.0 even when fewer than 10 jobs are correct.
- nDCG@10: binary gains, rewards correct jobs near the top.
- recall@100: share of all correct jobs in the top 100; only for queries with
  at most 100 correct jobs, None otherwise.

`compare` reports the paired difference between two runs with a bootstrap 95%
confidence interval over queries. A difference whose interval includes zero
is not a result.

    uv run python -m job_search.evaluate reference data/normalised/himalayas-2026-10-13.jsonl.gz
    uv run python -m job_search.evaluate score data/runs/oracle-2026-10-13.jsonl
    uv run python -m job_search.evaluate compare data/runs/random-2026-10-13.jsonl data/runs/oracle-2026-10-13.jsonl
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import random
import statistics
from pathlib import Path

from job_search.normalise import snapshot_date
from job_search.specs import answer_sets, ids_hash

DEPTH = 100
CUTOFF = 10
RECALL_MAX_RELEVANT = 100
METRICS = ["precision@10", "ndcg@10", "recall@100"]
BOOTSTRAP_SAMPLES = 10_000
SEED = 20261014
EVAL_DIR = Path("data/eval")
NORMALISED_DIR = Path("data/normalised")
RUNS_DIR = Path("data/runs")
RESULTS_DIR = Path("data/results")


# --- metrics --------------------------------------------------------------


def condense(ranking: list[str], ignored: set[str]) -> list[str]:
    """Drop ignored and repeated jobs, keep order."""
    seen, out = set(), []
    for job_id in ranking:
        if job_id in ignored or job_id in seen:
            continue
        seen.add(job_id)
        out.append(job_id)
    return out


def score_query(ranking: list[str], relevant: set[str], ignored: set[str]) -> dict:
    ranked = condense(ranking, ignored)
    hits = [job_id in relevant for job_id in ranked]
    top = hits[:CUTOFF]
    ideal = min(CUTOFF, len(relevant))
    dcg = sum(1 / math.log2(i + 2) for i, hit in enumerate(top) if hit)
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal))
    recall = sum(hits[:DEPTH]) / len(relevant) if len(relevant) <= RECALL_MAX_RELEVANT else None
    return {"precision@10": sum(top) / ideal, "ndcg@10": dcg / idcg, "recall@100": recall}


# --- data -----------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_answer_key(queries_path: Path, normalised_path: Path) -> tuple[list[dict], set[str]]:
    """Queries with their relevant and ignored sets, verified against the frozen counts."""
    jobs = load_jsonl(normalised_path)
    queries = load_jsonl(queries_path)
    for q in queries:
        relevant, ignored = answer_sets(jobs, q["constraints"])
        if len(relevant) != q["relevant"] or ids_hash(relevant) != q["relevant_hash"]:
            raise ValueError(f"{q['spec_id']}: answer set differs from the frozen one")
        q["relevant_ids"], q["ignored_ids"] = set(relevant), set(ignored)
    return queries, {job["id"] for job in jobs}


def default_paths(run_path: Path) -> tuple[Path, Path]:
    date = snapshot_date(run_path)
    return EVAL_DIR / f"queries-{date}.jsonl", NORMALISED_DIR / f"himalayas-{date}.jsonl.gz"


def load_run(run_path: Path, queries: list[dict], corpus: set[str]) -> dict[str, list[str]]:
    run = {row["spec_id"]: row["ranking"] for row in load_jsonl(run_path)}
    missing = [q["spec_id"] for q in queries if q["spec_id"] not in run]
    if missing:
        raise ValueError(f"run has no ranking for {len(missing)} queries, e.g. {missing[:3]}")
    for spec_id, ranking in run.items():
        if len(ranking) > DEPTH:
            raise ValueError(f"{spec_id}: ranking longer than {DEPTH}")
        unknown = set(ranking) - corpus
        if unknown:
            raise ValueError(f"{spec_id}: {len(unknown)} job ids not in the snapshot")
    return run


# --- scoring --------------------------------------------------------------


def score_run(run: dict[str, list[str]], queries: list[dict]) -> list[dict]:
    return [
        {
            "spec_id": q["spec_id"],
            "style": q["style"],
            "constraints": len(q["constraints"]),
            "fields": sorted(q["constraints"]),
            **score_query(run[q["spec_id"]], q["relevant_ids"], q["ignored_ids"]),
        }
        for q in queries
    ]


def mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(statistics.fmean(present), 4) if present else None


def summarise(per_query: list[dict]) -> dict:
    def block(rows):
        return {"queries": len(rows), **{m: mean([r[m] for r in rows]) for m in METRICS}}

    slices = collections.defaultdict(list)
    for r in per_query:
        slices[f"style={r['style']}"].append(r)
        slices[f"constraints={r['constraints']}"].append(r)
        for field in r["fields"]:
            slices[f"field={field}"].append(r)
    return {"overall": block(per_query), "slices": {k: block(v) for k, v in sorted(slices.items())}}


def paired_bootstrap(a: list[float], b: list[float], samples: int = BOOTSTRAP_SAMPLES,
                     seed: int = SEED) -> dict:
    """Mean of b - a with a 95% percentile interval, resampling queries."""
    diffs = [y - x for x, y in zip(a, b)]
    if not diffs:
        return {"difference": None, "ci95": None, "result": False, "queries": 0}
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(statistics.fmean(rng.choices(diffs, k=n)) for _ in range(samples))
    low, high = means[int(0.025 * samples)], means[int(0.975 * samples) - 1]
    return {"difference": round(statistics.fmean(diffs), 4), "ci95": [round(low, 4), round(high, 4)],
            "result": not (low <= 0 <= high), "queries": n}


def compare(per_query_a: list[dict], per_query_b: list[dict]) -> dict:
    b_by_id = {r["spec_id"]: r for r in per_query_b}
    out = {}
    for metric in METRICS:
        pairs = [(r[metric], b_by_id[r["spec_id"]][metric]) for r in per_query_a
                 if r[metric] is not None and b_by_id[r["spec_id"]][metric] is not None]
        out[metric] = paired_bootstrap([x for x, _ in pairs], [y for _, y in pairs])
    return out


# --- reference runs -------------------------------------------------------


def reference_runs(queries: list[dict], corpus: set[str], seed: int = SEED) -> dict[str, list[dict]]:
    """oracle: correct jobs first; random: a seeded shuffle of the whole snapshot."""
    rng = random.Random(seed)
    ids = sorted(corpus)
    oracle, shuffled = [], []
    for q in queries:
        oracle.append({"spec_id": q["spec_id"], "ranking": sorted(q["relevant_ids"])[:DEPTH]})
        shuffled.append({"spec_id": q["spec_id"], "ranking": rng.sample(ids, min(DEPTH, len(ids)))})
    return {"oracle": oracle, "random": shuffled}


# --- run ------------------------------------------------------------------


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def score_file(run_path: Path) -> tuple[list[dict], dict]:
    queries, corpus = load_answer_key(*default_paths(run_path))
    per_query = score_run(load_run(run_path, queries, corpus), queries)
    return per_query, summarise(per_query)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("reference").add_argument("normalised", type=Path)
    sub.add_parser("score").add_argument("run", type=Path)
    cmp = sub.add_parser("compare")
    cmp.add_argument("baseline", type=Path)
    cmp.add_argument("candidate", type=Path)
    args = parser.parse_args(argv)

    if args.command == "reference":
        date = snapshot_date(args.normalised)
        queries, corpus = load_answer_key(EVAL_DIR / f"queries-{date}.jsonl", args.normalised)
        for name, rows in reference_runs(queries, corpus).items():
            write_jsonl(RUNS_DIR / f"{name}-{date}.jsonl", rows)
            print(RUNS_DIR / f"{name}-{date}.jsonl")
    elif args.command == "score":
        per_query, summary = score_file(args.run)
        name = args.run.name.removesuffix(".jsonl")
        write_jsonl(RESULTS_DIR / f"{name}.per-query.jsonl", per_query)
        (RESULTS_DIR / f"{name}.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary["overall"], indent=2))
    else:
        a, _ = score_file(args.baseline)
        b, _ = score_file(args.candidate)
        print(json.dumps({"baseline": args.baseline.name, "candidate": args.candidate.name, **compare(a, b)},
                         indent=2))


if __name__ == "__main__":
    main()
