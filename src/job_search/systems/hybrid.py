"""Hybrid baseline: BM25 and embeddings combined by reciprocal rank fusion.

Each job scores sum(1 / (K + rank)) over the runs it appears in, with the
conventional K = 60 and no tuning. It fuses the two existing run files (top
100 each), so it costs nothing.

    uv run python -m job_search.systems.hybrid data/normalised/himalayas-2026-10-13.jsonl.gz
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from job_search.evaluate import DEPTH, RUNS_DIR
from job_search.normalise import snapshot_date
from job_search.systems import add_arguments, run_name, write_run

K = 60
SOURCES = ["bm25", "embeddings"]


def fuse(rankings: list[list[str]], k: int = K, depth: int = DEPTH) -> list[str]:
    scores: dict[str, float] = collections.defaultdict(float)
    for ranking in rankings:
        for rank, job_id in enumerate(ranking, 1):
            scores[job_id] += 1 / (k + rank)
    # Ties broken by job id so the output is deterministic.
    return sorted(scores, key=lambda j: (-scores[j], j))[:depth]


def load_run(path: Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8") as f:
        return {row["spec_id"]: row["ranking"] for row in map(json.loads, f)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(parser)
    args = parser.parse_args(argv)

    date = snapshot_date(args.normalised)
    runs = [load_run(RUNS_DIR / f"{run_name(name, args.queries)}-{date}.jsonl") for name in SOURCES]
    rankings = {spec_id: fuse([run[spec_id] for run in runs]) for spec_id in runs[0]}
    print(write_run(run_name("hybrid", args.queries), args.normalised, rankings))


if __name__ == "__main__":
    main()
