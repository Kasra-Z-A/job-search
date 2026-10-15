"""Answer one message of a job search conversation.

1. Understand: the first message goes through the evaluated parser
   (`systems.parse`); follow-ups through `followup`, which also rewrites the
   request as one standalone search sentence.
2. Retrieve: filter on the fields and rank by embedding similarity, as the
   metadata-track system does (`systems.filtered`); keep the top CANDIDATES.
3. Answer: Claude Haiku 5.5 reads the candidates and picks the best fits with
   reasons (`answer`); `answer.ground` checks the text against the data. If
   the answer step fails, the ranking alone is shown.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np

from job_search.app import answer, followup
from job_search.specs import check
from job_search.systems.filtered import filter_outcome, rank_filtered
from job_search.systems.parse import as_constraints, parse_one, schema

CANDIDATES = 20
MAX_QUERY_CHARS = 200


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
    def __init__(self, bundle_dir: Path, anthropic_client, openai_client):
        self.anthropic = anthropic_client
        self.openai = openai_client
        with gzip.open(bundle_dir / "jobs.json.gz", "rt", encoding="utf-8") as f:
            self.jobs = json.load(f)
        self.vectors = np.load(bundle_dir / "vectors.npy").astype(np.float32)
        self.texts_path = bundle_dir / "texts.bin"
        self.text_offsets = np.load(bundle_dir / "text_offsets.npy")
        self.position = {job["id"]: i for i, job in enumerate(self.jobs)}
        values = json.loads((bundle_dir / "values.json").read_text())
        self.snapshot = values["snapshot"]
        self.embedding_model = values["embedding_model"]
        self.schema = schema(values["categories"], values["countries"], values["employment_types"])
        self.followup_schema = followup.schema(self.schema)

    def text(self, job_id: str) -> str:
        """One job's full description, read from disk on demand."""
        i = self.position[job_id]
        start, end = int(self.text_offsets[i]), int(self.text_offsets[i + 1])
        with self.texts_path.open("rb") as f:
            f.seek(start)
            return f.read(end - start).decode("utf-8")

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

    def retrieve(self, understood: dict, k: int = CANDIDATES) -> tuple[list[dict], dict, dict]:
        filters = as_constraints(understood["filters"])
        keep = [i for i, j in enumerate(self.jobs)
                if not understood["require_salary"] or j["salary_status"] == "stated"]
        jobs = [self.jobs[i] for i in keep]
        similarity = (self.vectors[keep] @ self.embed(understood["search_text"]))
        outcomes = [filter_outcome(job, filters) for job in jobs]
        ranking = rank_filtered(jobs, similarity, filters, k)
        counts = {"match": outcomes.count("match"), "maybe": outcomes.count("unknown"), "total": len(self.jobs)}
        candidates = [{"job": self.jobs[self.position[j]], "meets": explain(self.jobs[self.position[j]], filters),
                       "text": self.text(j)} for j in ranking]
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

        understood = timed("understand", self.understand, message, history or [])
        candidates, counts, _ = timed("retrieve", self.retrieve, understood)
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
