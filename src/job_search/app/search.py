"""Answer one search query for the demo, as the metadata-track system does.

Claude Haiku 5.5 turns the query into filters, OpenAI embeds it, and jobs are
filtered on their fields and ranked by similarity (`systems.filtered`). Each
result says which requested filters it meets and which it may meet (field not
stated), so the page can show why it matched.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np

from job_search.specs import check
from job_search.systems.filtered import filter_outcome, rank_filtered
from job_search.systems.parse import as_constraints, parse_one, schema

RESULTS = 20
MAX_QUERY_CHARS = 200


class Searcher:
    def __init__(self, bundle_dir: Path, anthropic_client, openai_client):
        self.anthropic = anthropic_client
        self.openai = openai_client
        with gzip.open(bundle_dir / "jobs.json.gz", "rt", encoding="utf-8") as f:
            self.jobs = json.load(f)
        self.vectors = np.load(bundle_dir / "vectors.npy").astype(np.float32)
        values = json.loads((bundle_dir / "values.json").read_text())
        self.snapshot = values["snapshot"]
        self.embedding_model = values["embedding_model"]
        self.schema = schema(values["categories"], values["countries"], values["employment_types"])

    def embed(self, query: str) -> np.ndarray:
        response = self.openai.embeddings.create(model=self.embedding_model, input=[query])
        return np.asarray(response.data[0].embedding, dtype=np.float32)

    def search(self, query: str, k: int = RESULTS) -> dict:
        times = {}
        start = time.perf_counter()
        parsed = parse_one(self.anthropic, query, self.schema)["parsed"]
        times["understand_ms"] = round((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        similarity = self.vectors @ self.embed(query)
        times["embed_ms"] = round((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        filters = as_constraints(parsed)
        outcomes = [filter_outcome(job, filters) for job in self.jobs]
        ranking = rank_filtered(self.jobs, similarity, filters, k)
        by_id = {job["id"]: job for job in self.jobs}
        times["rank_ms"] = round((time.perf_counter() - start) * 1000)

        return {
            "query": query,
            "snapshot": self.snapshot,
            "filters": parsed,
            "counts": {"match": outcomes.count("match"), "maybe": outcomes.count("unknown"),
                       "total": len(self.jobs)},
            "results": [{**by_id[job_id], "meets": explain(by_id[job_id], filters)} for job_id in ranking],
            "timing": times,
        }


def explain(job: dict, filters: dict) -> dict:
    """Per requested field: 'yes', or 'unknown' when the posting does not state it."""
    out = {}
    for field, values in filters.items():
        if values in (None, []):
            continue
        outcomes = {check(job, field, v) for v in (values if isinstance(values, list) else [values])}
        out[field] = "yes" if "yes" in outcomes else "unknown"
    return out
