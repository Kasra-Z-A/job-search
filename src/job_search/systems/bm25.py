"""Keyword baseline: BM25 over job title and description.

Tokens are lowercase words and numbers, English stopwords removed, stemmed
with the Snowball English stemmer so "engineer" matches "engineering". The
query is used as typed. BM25 sees only text, so salary thresholds and
"open to people in X" match only where the same words appear in a posting;
that gap is what later systems are measured against.

    uv run python -m job_search.systems.bm25 data/normalised/himalayas-2026-10-13.jsonl.gz
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import snowballstemmer
from rank_bm25 import BM25Okapi

from job_search.evaluate import DEPTH
from job_search.systems import job_text, load_jobs, load_queries, write_run

STOPWORDS = set("""
a an and are as at be but by for from has have i i'm im in is it its me my of on or our
so that the their them they this to us was we were what which who will with you your
any anything some there got hey can find show looking want""".split())
_stemmer = snowballstemmer.stemmer("english")
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    words = [w for w in _TOKEN.findall(text.lower()) if w not in STOPWORDS]
    return _stemmer.stemWords(words)


class BM25Search:
    def __init__(self, jobs: list[dict]):
        self.ids = [job["id"] for job in jobs]
        self.index = BM25Okapi([tokenize(job_text(job)) for job in jobs])

    def search(self, query: str, k: int = DEPTH) -> list[str]:
        scores = self.index.get_scores(tokenize(query))
        top = np.argsort(-scores, kind="stable")[:k]
        return [self.ids[i] for i in top]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("normalised", type=Path)
    args = parser.parse_args(argv)

    engine = BM25Search(load_jobs(args.normalised))
    queries = load_queries(args.normalised)
    rankings = {}
    for n, q in enumerate(queries, 1):
        rankings[q["spec_id"]] = engine.search(q["query"])
        if n % 250 == 0:
            print(f"  {n}/{len(queries)} queries", file=sys.stderr)
    print(write_run("bm25", args.normalised, rankings))


if __name__ == "__main__":
    main()
