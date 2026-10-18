"""Pack a snapshot for the demo backend, and the evaluation results for the page.

Writes `data/app/<date>/` (gitignored, shipped inside the Lambda package):

- jobs.json.gz: the fields search and display need, plus a short excerpt;
- texts.bin and text_offsets.npy: the full descriptions, concatenated, and
  where each starts. The answer step reads only its candidates' descriptions,
  so starting a container does not parse ~40 MB of text;
- vectors.npy: job embeddings as float16, half the size, ranking unchanged
  in practice;
- values.json: allowed categories, countries and employment types for the
  query parser's schema.

and `web/results.json` (committed): the scores the page charts.

    uv run python -m job_search.app.bundle data/normalised/himalayas-2026-10-13.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from job_search.normalise import snapshot_date
from job_search.systems import load_jobs
from job_search.systems.embeddings import CACHE_DIR
from job_search.systems.embeddings import MODEL as EMBEDDING_MODEL
from job_search.systems.parse import field_values

APP_DIR = Path("data/app")
WEB_RESULTS = Path("web/results.json")
EXCERPT_CHARS = 280
KEEP = ["id", "title", "company", "published_at", "seniority", "categories", "countries", "worldwide",
        "employment_type", "salary_status", "salary_min_usd", "salary_max_usd"]
JUDGED_SET = "topical-v2"
# Systems shown on the page, in order, with the label a visitor reads. Scores
# come from the LLM judge on the topical queries (`job_search.judge`); the
# judge checks ran once, on the first 73 queries' grades.
SYSTEMS = [
    ("bm25", "Keyword search (BM25)"),
    ("hybrid", "Hybrid search (BM25 + embeddings)"),
    ("embeddings", "Semantic search (embeddings)"),
    ("textonly", "Filtered semantic search, labels extracted from text"),
    ("filtered", "Filtered semantic search, Himalayas labels"),
]


def excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit].rsplit(" ", 1)[0] + "…"


def slim(job: dict) -> dict:
    return {**{k: job[k] for k in KEEP}, "excerpt": excerpt(job["text"])}


def results_summary(date: str, results_dir: Path = Path("data/results")) -> dict:
    judged = json.loads((results_dir / f"judged.{JUDGED_SET}-{date}.json").read_text())
    checks = json.loads((results_dir / f"judge-checks.topical-{date}.json").read_text())
    return {
        "snapshot": date,
        "queries": judged["queries"],
        "pairs_graded": judged["pairs_graded"],
        "systems": [{"name": name, "label": label, **judged["systems"][name]} for name, label in SYSTEMS],
        "checks": {"second_judge_kappa": checks["haiku_vs_second"]["weighted_kappa"],
                   "second_judge_exact": checks["haiku_vs_second"]["exact_agreement"],
                   "blind_kappa": checks["haiku_vs_blind"]["weighted_kappa"],
                   "random_pairs_zero": checks["random_pairs"]["grades"].get("0", 0) / checks["random_pairs"]["pairs"]},
    }


def build(normalised: Path, app_dir: Path = APP_DIR) -> Path:
    date = snapshot_date(normalised)
    jobs = load_jobs(normalised)
    cached = np.load(CACHE_DIR / f"{EMBEDDING_MODEL}-jobs-{date}.npz", allow_pickle=False)
    if cached["ids"].tolist() != [j["id"] for j in jobs]:
        raise ValueError("embedding cache does not match the snapshot")

    out = app_dir / date
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "jobs.json.gz", "wt", encoding="utf-8") as f:
        json.dump([slim(j) for j in jobs], f, ensure_ascii=False)
    np.save(out / "vectors.npy", cached["vectors"].astype(np.float16))
    encoded = [j["text"].encode("utf-8") for j in jobs]
    (out / "texts.bin").write_bytes(b"".join(encoded))
    np.save(out / "text_offsets.npy", np.cumsum([0, *map(len, encoded)], dtype=np.int64))
    categories, countries, employment_types = field_values(jobs)
    (out / "values.json").write_text(json.dumps({
        "snapshot": date, "embedding_model": EMBEDDING_MODEL, "categories": categories,
        "countries": countries, "employment_types": employment_types}, indent=1))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("normalised", type=Path)
    args = parser.parse_args(argv)
    print(build(args.normalised))
    WEB_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    WEB_RESULTS.write_text(json.dumps(results_summary(snapshot_date(args.normalised)), indent=1) + "\n")
    print(WEB_RESULTS)


if __name__ == "__main__":
    main()
