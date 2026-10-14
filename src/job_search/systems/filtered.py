"""Main system: LLM-parsed filters on the structured fields, then embeddings.

For each query, the parsed filters (see `parse`) are applied to the
normalised fields with the matching rules defined in `specs` (the product's
definition of "at least X", "open to people in X" and the seniority levels):

- a job that contradicts a filter is dropped;
- a job that satisfies every filter ranks first; a job whose field is unknown
  (no stated salary, no category) stays, ranked after the known matches;
- within each group, jobs are ordered by embedding similarity to the query.

This is the metadata track: the system reads structured fields, unlike the
text-only baselines. Parses and embeddings are read from their caches, so
building the run costs nothing.

    uv run python -m job_search.systems.filtered data/normalised/himalayas-2026-10-13.jsonl.gz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from job_search.evaluate import DEPTH
from job_search.normalise import snapshot_date
from job_search.specs import check
from job_search.systems import add_arguments, load_jobs, load_queries, run_name, write_run
from job_search.systems.embeddings import CACHE_DIR
from job_search.systems.embeddings import MODEL as EMBEDDING_MODEL
from job_search.systems.parse import as_constraints, parses_path


def filter_outcome(job: dict, filters: dict) -> str:
    """'match', 'unknown' or 'no'. Several values in one field mean any of them."""
    unknown = False
    for field, values in filters.items():
        if values in (None, []):
            continue
        outcomes = {check(job, field, v) for v in (values if isinstance(values, list) else [values])}
        if "yes" in outcomes:
            continue
        if "unknown" in outcomes:
            unknown = True
        else:
            return "no"
    return "unknown" if unknown else "match"


def rank_filtered(jobs: list[dict], similarity: np.ndarray, filters: dict, k: int = DEPTH) -> list[str]:
    tiers = {"match": 0, "unknown": 1}
    keep = [(tiers[o], -similarity[i], jobs[i]["id"])
            for i, job in enumerate(jobs) if (o := filter_outcome(job, filters)) != "no"]
    return [job_id for _, _, job_id in sorted(keep)[:k]]


def load_vectors(path: Path, ids: list[str]) -> np.ndarray:
    cached = np.load(path, allow_pickle=False)
    if cached["ids"].tolist() != ids:
        raise ValueError(f"{path} does not match the current ids")
    return cached["vectors"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(parser)
    args = parser.parse_args(argv)

    date = snapshot_date(args.normalised)
    jobs = load_jobs(args.normalised)
    queries = load_queries(args.normalised, args.queries)
    job_vectors = load_vectors(CACHE_DIR / f"{EMBEDDING_MODEL}-jobs-{date}.npz", [j["id"] for j in jobs])
    query_vectors = load_vectors(CACHE_DIR / f"{EMBEDDING_MODEL}-{args.queries}-{date}.npz",
                                 [q["spec_id"] for q in queries])
    with parses_path(args.normalised, args.queries).open() as f:
        parses = {r["spec_id"]: r["parsed"] for r in map(json.loads, f)}

    similarity = query_vectors @ job_vectors.T
    rankings = {
        q["spec_id"]: rank_filtered(jobs, similarity[n], as_constraints(parses[q["spec_id"]]))
        for n, q in enumerate(queries)
    }
    print(write_run(run_name("filtered", args.queries), args.normalised, rankings))


if __name__ == "__main__":
    main()
