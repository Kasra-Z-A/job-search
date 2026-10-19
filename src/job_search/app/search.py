"""Answer one message of a job search conversation.

1. Understand: the first message goes through the evaluated parser
   (`systems.parse`); follow-ups through `followup`, which also rewrites the
   request as one standalone search sentence.
2. Retrieve: filter on the fields and rank by embedding similarity, as the
   metadata-track system does (`systems.filtered`); keep the top CANDIDATES.
3. Answer: Claude Haiku 5.5 reads the candidates and picks the best fits with
   reasons (`answer`); `answer.ground` checks the text against the data. If
   the answer step fails, the ranking alone is shown.

A first message is embedded as typed, so its embedding runs alongside the
parser instead of after it. Descriptions are read only for the candidates:
from a local file, or with S3 range requests in parallel (`S3Texts`), so a
cold start does not download them all.
"""

from __future__ import annotations

import gzip
import json
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from job_search.app import answer, followup
from job_search.specs import check
from job_search.systems.filtered import filter_outcome
from job_search.systems.parse import as_constraints, parse_one, schema

CANDIDATES = 20
MAX_QUERY_CHARS = 200
TEXT_CACHE = 2000  # descriptions kept in memory, ~10 MB


class FileTexts:
    """Descriptions from a local texts.bin, by byte range."""

    def __init__(self, path: Path):
        self.path = path

    def read(self, ranges: list[tuple[int, int]]) -> list[bytes]:
        with self.path.open("rb") as f:
            out = []
            for start, end in ranges:
                f.seek(start)
                out.append(f.read(end - start))
            return out


class S3Texts:
    """Descriptions read straight from the bundle's texts.bin in S3, in parallel."""

    def __init__(self, client, bucket: str, key: str, workers: int = CANDIDATES):
        self.client, self.bucket, self.key = client, bucket, key
        self.pool = ThreadPoolExecutor(workers)

    def get(self, start: int, end: int) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=self.key, Range=f"bytes={start}-{end - 1}")
        return response["Body"].read()

    def read(self, ranges: list[tuple[int, int]]) -> list[bytes]:
        return list(self.pool.map(lambda r: self.get(*r) if r[1] > r[0] else b"", ranges))


def understood_list(parsed: dict, require_salary: bool = False) -> list[str]:
    chips = [*parsed["seniority"], *parsed["categories"], *(f"open to {c}" for c in parsed["countries"]),
             *parsed["employment_types"]]
    if parsed["salary_min_usd"]:
        chips.append(f"at least ${parsed['salary_min_usd']:,} a year")
    if require_salary:
        chips.append("salary stated")
    return chips


def explain(job: dict, filters: dict) -> dict:
    """Per requested field: 'yes', or 'unknown' when the posting does not state it."""
    out = {}
    for field, values in filters.items():
        if values in (None, []):
            continue
        outcomes = {check(job, field, v) for v in (values if isinstance(values, list) else [values])}
        out[field] = "yes" if "yes" in outcomes else "unknown"
    return out


class Searcher:
    def __init__(self, bundle_dir: Path, anthropic_client, openai_client, texts=None):
        self.anthropic = anthropic_client
        self.openai = openai_client
        with gzip.open(bundle_dir / "jobs.json.gz", "rt", encoding="utf-8") as f:
            self.jobs = json.load(f)
        self.vectors = np.load(bundle_dir / "vectors.npy").astype(np.float32)
        self.texts = texts or FileTexts(bundle_dir / "texts.bin")
        self.text_cache: OrderedDict[str, str] = OrderedDict()
        self.text_offsets = np.load(bundle_dir / "text_offsets.npy")
        self.pool = ThreadPoolExecutor(2)
        self.position = {job["id"]: i for i, job in enumerate(self.jobs)}
        values = json.loads((bundle_dir / "values.json").read_text())
        self.snapshot = values["snapshot"]
        self.embedding_model = values["embedding_model"]
        self.schema = schema(values["categories"], values["countries"], values["employment_types"])
        self.followup_schema = followup.schema(self.schema)

    def text_many(self, job_ids: list[str]) -> list[str]:
        """Full descriptions, read on demand; recent ones are kept in memory."""
        missing = [j for j in dict.fromkeys(job_ids) if j not in self.text_cache]
        ranges = [(int(self.text_offsets[self.position[j]]), int(self.text_offsets[self.position[j] + 1]))
                  for j in missing]
        for job_id, data in zip(missing, self.texts.read(ranges) if missing else []):
            self.text_cache[job_id] = data.decode("utf-8")
        for job_id in job_ids:
            self.text_cache.move_to_end(job_id)
        while len(self.text_cache) > TEXT_CACHE:
            self.text_cache.popitem(last=False)
        return [self.text_cache[j] for j in job_ids]

    def text(self, job_id: str) -> str:
        return self.text_many([job_id])[0]

    def embed(self, text: str) -> np.ndarray:
        response = self.openai.embeddings.create(model=self.embedding_model, input=[text])
        return np.asarray(response.data[0].embedding, dtype=np.float32)

    # --- steps -------------------------------------------------------------

    def understand(self, message: str, history: list[dict]) -> dict:
        if not history:
            parsed = parse_one(self.anthropic, message, self.schema)["parsed"]
            return {"filters": parsed, "search_text": message, "require_salary": False}
        turns = history[-followup.MAX_TURNS:]
        titles = [[self.jobs[self.position[j]]["title"] for j in t["picks"] if j in self.position] for t in turns]
        cleaned = [{"q": t["q"][:MAX_QUERY_CHARS], "filters": followup.clean_filters(t["filters"], self.schema)}
                   for t in turns]
        out = answer.parse_output(self.anthropic.messages.create(
            **followup.request(cleaned, titles, message, self.followup_schema)))
        return {"filters": followup.clean_filters(out, self.schema),
                "search_text": out["search_text"].strip() or message,
                "require_salary": bool(out["require_salary"])}

    def retrieve(self, understood: dict, query_vector: np.ndarray | None = None,
                 k: int = CANDIDATES) -> tuple[list[dict], dict, dict]:
        filters = as_constraints(understood["filters"])
        if query_vector is None:
            query_vector = self.embed(understood["search_text"])
        # One pass over all jobs: similarity for every job, outcomes only where needed.
        similarity = self.vectors @ query_vector
        tiers = {"match": 0, "unknown": 1}
        keep, counts = [], {"match": 0, "maybe": 0, "total": len(self.jobs)}
        for i, job in enumerate(self.jobs):
            if understood["require_salary"] and job["salary_status"] != "stated":
                continue
            outcome = filter_outcome(job, filters)
            if outcome == "no":
                continue
            counts["match" if outcome == "match" else "maybe"] += 1
            keep.append((tiers[outcome], -similarity[i], job["id"]))
        ranking = [job_id for _, _, job_id in sorted(keep)[:k]]
        texts = self.text_many(ranking)
        candidates = [{"job": self.jobs[self.position[j]], "meets": explain(self.jobs[self.position[j]], filters),
                       "text": text} for j, text in zip(ranking, texts)]
        return candidates, counts, filters

    def write_answer(self, request_text: str, understood: dict, counts: dict, candidates: list[dict]) -> dict:
        if not candidates:
            return {"summary": "No job meets all of that. Try loosening one part.", "picks": [], "problems": []}
        try:
            output = answer.parse_output(self.anthropic.messages.create(**answer.request(
                request_text, understood_list(understood["filters"], understood["require_salary"]),
                counts, candidates)))
        except Exception as exc:  # the ranking alone is still a useful answer
            print(f"answer failed: {type(exc).__name__}: {exc}")
            return {"summary": answer.FALLBACK_SUMMARY, "problems": ["answer failed"],
                    "picks": [{"index": i, "reason": ""} for i in range(min(answer.MAX_PICKS, len(candidates)))]}
        summary, picks, problems = answer.ground(output, candidates, understood["filters"]["salary_min_usd"])
        return {"summary": summary, "picks": picks, "problems": problems}

    # --- one message --------------------------------------------------------

    def search(self, message: str, history: list[dict] | None = None) -> dict:
        times = {}

        def timed(name, fn, *args):
            start = time.perf_counter()
            result = fn(*args)
            times[f"{name}_ms"] = round((time.perf_counter() - start) * 1000)
            return result

        # A first message is embedded as typed, so that call need not wait for the parser.
        early = None if history else self.pool.submit(timed, "embed", self.embed, message)
        understood = timed("understand", self.understand, message, history or [])
        query_vector = early.result() if early and understood["search_text"] == message else None
        candidates, counts, _ = timed("retrieve", self.retrieve, understood, query_vector)
        written = timed("answer", self.write_answer, understood["search_text"], understood, counts, candidates)

        picked = {p["index"] for p in written["picks"]}
        card = lambda c: {**c["job"], "meets": c["meets"]}  # noqa: E731
        return {
            "query": message,
            "snapshot": self.snapshot,
            "filters": understood["filters"],
            "require_salary": understood["require_salary"],
            "search_text": understood["search_text"],
            "counts": counts,
            "summary": written["summary"],
            "picks": [{**card(candidates[p["index"]]), "reason": p["reason"]} for p in written["picks"]],
            "more": [card(c) for i, c in enumerate(candidates) if i not in picked],
            "grounding": written["problems"],
            "timing": times,
        }
