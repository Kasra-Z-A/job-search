"""Search systems. Each one ranks the snapshot's jobs for every evaluation
query and writes a run file for `job_search.evaluate`.

Query sets live in `data/eval/<set>-<date>.jsonl`: `queries` is the main
set; others (such as `sloppy`) reuse its specs with different wording. A run
on another set is named `<system>.<set>-<date>.jsonl`."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from job_search.evaluate import DEPTH, RUNS_DIR
from job_search.normalise import snapshot_date


def load_jobs(normalised: Path) -> list[dict]:
    with gzip.open(normalised, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


MAIN_SET = "queries"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("normalised", type=Path)
    parser.add_argument("--queries", default=MAIN_SET, help="query set name in data/eval/")


def run_name(system: str, query_set: str) -> str:
    return system if query_set == MAIN_SET else f"{system}.{query_set}"


def load_queries(normalised: Path, query_set: str = MAIN_SET) -> list[dict]:
    path = Path("data/eval") / f"{query_set}-{snapshot_date(normalised)}.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def job_text(job: dict) -> str:
    """What a system sees of a job: title and description, no structured fields."""
    return f"{job['title']}\n{job['text']}"


def write_run(name: str, normalised: Path, rankings: dict[str, list[str]], runs_dir: Path = RUNS_DIR) -> Path:
    path = runs_dir / f"{name}-{snapshot_date(normalised)}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for spec_id, ranking in rankings.items():
            f.write(json.dumps({"spec_id": spec_id, "ranking": ranking[:DEPTH]}) + "\n")
    return path
