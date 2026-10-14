"""Semantic baseline: OpenAI embeddings, ranked by cosine similarity.

Jobs (title and description, first MAX_CHARS characters) and queries are
embedded with text-embedding-3-small; jobs are ranked by dot product, which
equals cosine similarity because the vectors come back normalised. Like BM25
it sees only text.

Embeddings are cached in `data/embeddings/` (gitignored), so a rerun costs
nothing; a cache is reused only if its ids match the jobs or queries exactly.
At $0.02 per million tokens the snapshot (~10.6M tokens) costs about $0.21.
The key comes from OPENAI_API_KEY.

    uv run python -m job_search.systems.embeddings data/normalised/himalayas-2026-10-13.jsonl.gz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

from job_search.evaluate import DEPTH
from job_search.normalise import snapshot_date
from job_search.systems import add_arguments, job_text, load_jobs, load_queries, run_name, write_run

MODEL = "text-embedding-3-small"
PRICE_PER_MILLION = 0.02
MAX_CHARS = 20_000  # well under the model's 8,191-token input limit
BATCH = 64  # inputs per request, far below the per-request token limit
CACHE_DIR = Path("data/embeddings")


def embed(client, texts: list[str], batch: int = BATCH) -> tuple[np.ndarray, int]:
    """Vectors in input order and the tokens used."""
    vectors, tokens = [], 0
    for start in range(0, len(texts), batch):
        response = client.embeddings.create(model=MODEL, input=texts[start:start + batch])
        vectors += [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
        tokens += response.usage.total_tokens
        if (start // batch) % 20 == 0:
            print(f"  embedded {start + len(texts[start:start + batch])}/{len(texts)}", file=sys.stderr)
    return np.asarray(vectors, dtype=np.float32), tokens


def cached_embed(client, ids: list[str], texts: list[str], path: Path) -> tuple[np.ndarray, int]:
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        if cached["ids"].tolist() == ids:
            return cached["vectors"], 0
        raise ValueError(f"{path} holds different ids; delete it to re-embed")
    vectors, tokens = embed(client, texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.asarray(ids), vectors=vectors)
    return vectors, tokens


def rank(job_vectors: np.ndarray, query_vectors: np.ndarray, job_ids: list[str], k: int = DEPTH) -> list[list[str]]:
    scores = query_vectors @ job_vectors.T
    top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return [[job_ids[i] for i in row] for row in top]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(parser)
    args = parser.parse_args(argv)

    date = snapshot_date(args.normalised)
    jobs = load_jobs(args.normalised)
    queries = load_queries(args.normalised, args.queries)
    client = OpenAI()

    job_ids = [j["id"] for j in jobs]
    job_vectors, job_tokens = cached_embed(
        client, job_ids, [job_text(j)[:MAX_CHARS] for j in jobs], CACHE_DIR / f"{MODEL}-jobs-{date}.npz")
    query_vectors, query_tokens = cached_embed(
        client, [q["spec_id"] for q in queries], [q["query"] for q in queries],
        CACHE_DIR / f"{MODEL}-{args.queries}-{date}.npz")
    tokens = job_tokens + query_tokens
    print(f"embedded {tokens:,} new tokens, ${tokens * PRICE_PER_MILLION / 1e6:.2f}", file=sys.stderr)

    rankings = dict(zip([q["spec_id"] for q in queries], rank(job_vectors, query_vectors, job_ids)))
    print(write_run(run_name("embeddings", args.queries), args.normalised, rankings))


if __name__ == "__main__":
    main()
