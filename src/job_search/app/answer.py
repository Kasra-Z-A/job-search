"""Write the answer: a short summary and the best-fitting jobs, with reasons.

Claude Haiku 5.5 reads the retrieved candidates (their fields, whether they
meet each requirement, and the start of each description) and returns a
summary of at most three sentences and up to MAX_PICKS jobs by number, each
with a one-line reason. The page renders the cards from the dataset, so
links, salaries and locations never come from the model.

`ground` then checks the model's text: picks must be candidates, each at most
once, and every dollar amount in the summary or a reason must be one the data
or the request contains. Text that fails is replaced or dropped, and the
violation is reported.
"""

from __future__ import annotations

import json
import re

from job_search.systems.parse import MODEL

MAX_PICKS = 6
DESCRIPTION_CHARS = 1200
MAX_TOKENS = 6000
FALLBACK_SUMMARY = "Here are the closest matches."

SYSTEM = """\
You help someone find a job on a remote job board. You get their request, the
filters it was understood as, how many jobs meet them, and the top candidate
jobs, numbered, each with its facts and the start of its description.

Pick the jobs worth applying to, best first, at most 6. Fewer strong picks
are better than a full list: leave out any job you would call a loose match,
such as a gig whose description shows a different kind of work than the
person wants. Prefer jobs that meet every requirement; a job marked "not
stated" for a requirement may be picked if it fits well otherwise, and its
reason says what is unknown.

Write a summary of at most two short sentences: how many good options there
are and what stands out. Give each pick a reason of at most 20 words.

Use only facts given here. Never invent a salary, location or company
detail. Do not include links."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "picks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"job": {"type": "integer"}, "reason": {"type": "string"}},
                "required": ["job", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "picks"],
    "additionalProperties": False,
}

FIELD_NAMES = {"seniority": "level", "category": "job family", "country": "location",
               "employment_type": "contract", "salary_min_usd": "salary"}


def salary_text(job: dict) -> str:
    if job["salary_status"] != "stated":
        return "not stated"
    lo, hi = job["salary_min_usd"], job["salary_max_usd"]
    return f"${lo:,} a year" if lo == hi else f"${lo:,} to ${hi:,} a year"


def describe_candidate(n: int, job: dict, meets: dict, text: str) -> str:
    where = "worldwide" if job["worldwide"] else ", ".join(job["countries"][:8])
    met = ", ".join(f"{FIELD_NAMES[f]}: {'meets' if v == 'yes' else 'not stated'}" for f, v in meets.items())
    return (f"[{n}] {job['title']} at {job['company']}\n"
            f"Level: {', '.join(job['seniority']) or 'not stated'} | Family: {', '.join(job['categories']) or 'not stated'}"
            f" | Location: {where} | Contract: {job['employment_type'] or 'not stated'} | Pay: {salary_text(job)}\n"
            f"Requirements: {met or 'none given'}\n"
            f"Description: {' '.join(text[:DESCRIPTION_CHARS].split())}")


def request(request_text: str, understood: list[str], counts: dict, candidates: list[dict]) -> dict:
    listing = "\n\n".join(describe_candidate(n, c["job"], c["meets"], c["text"]) for n, c in enumerate(candidates, 1))
    prompt = (f"Request: {request_text}\n"
              f"Understood as: {', '.join(understood) or 'no filters'}\n"
              f"Jobs meeting every requirement: {counts['match']}; possibly meeting (detail not stated): "
              f"{counts['maybe']}; of {counts['total']} jobs in total.\n\n"
              f"Candidates:\n\n{listing}")
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM,
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": ANSWER_SCHEMA}},
        "messages": [{"role": "user", "content": prompt}],
    }


# --- grounding -----------------------------------------------------------

_MONEY = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s?([kK])?|\b(\d{2,3}(?:\.\d)?)\s?[kK]\b")


def amounts(text: str) -> list[int]:
    out = []
    for m in _MONEY.finditer(text):
        if m.group(3):
            out.append(round(float(m.group(3)) * 1000))
        else:
            value = float(m.group(1).replace(",", ""))
            out.append(round(value * 1000) if m.group(2) else round(value))
    return out


def known_amounts(candidates: list[dict], salary_filter: int) -> list[int]:
    values = [salary_filter] if salary_filter else []
    for c in candidates:
        if c["job"]["salary_status"] == "stated":
            values += [c["job"]["salary_min_usd"], c["job"]["salary_max_usd"]]
    return values


def grounded(text: str, known: list[int]) -> bool:
    """Every amount is within 1% or $1,000 of a known one ("$92k" for 91,742)."""
    return all(any(abs(a - k) <= max(1000, 0.01 * k) for k in known) for a in amounts(text))


def ground(output: dict, candidates: list[dict], salary_filter: int) -> tuple[str, list[dict], list[str]]:
    """Summary and picks that pass the checks, and what was removed."""
    known = known_amounts(candidates, salary_filter)
    problems = []
    summary = output["summary"].strip()
    if not grounded(summary, known):
        problems.append("summary amount")
        summary = FALLBACK_SUMMARY

    picks, seen = [], set()
    for pick in output["picks"]:
        n = pick["job"]
        if not 1 <= n <= len(candidates):
            problems.append("pick outside candidates")
            continue
        if n in seen:
            problems.append("repeated pick")
            continue
        seen.add(n)
        reason = pick["reason"].strip()
        if not grounded(reason, known):
            problems.append("reason amount")
            reason = ""
        picks.append({"index": n - 1, "reason": reason})
    return summary, picks[:MAX_PICKS], problems


def parse_output(message) -> dict:
    if message.stop_reason != "end_turn":
        raise ValueError(f"stop_reason {message.stop_reason}")
    return json.loads(next(block.text for block in message.content if block.type == "text"))
