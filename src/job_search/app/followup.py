"""Understand a follow-up message in a job search conversation.

The first message of a conversation goes through `systems.parse`, the
evaluated parser, unchanged. Later messages ("only full-time", "what about
Canada instead", "more like the second one") only make sense with what came
before, so Claude Haiku 5.5 gets the earlier turns (message, filters in force,
jobs shown) and returns:

- the complete filters that apply now, earlier ones kept unless changed;
- search_text: one standalone sentence of everything the user wants now,
  which meaning-based search embeds instead of the bare follow-up;
- require_salary: whether to keep only jobs that state their pay.

Follow-ups are not part of the evaluation.
"""

from __future__ import annotations

import json

from job_search.systems.parse import MAX_TOKENS, MODEL
from job_search.systems.parse import SYSTEM as PARSE_SYSTEM

MAX_TURNS = 4

SYSTEM = PARSE_SYSTEM + """

This message continues a conversation. You get the earlier turns: what the
user wrote, the filters in force after it, and the jobs that were shown,
numbered. Return the complete filters that should apply now: keep earlier
filters unless the new message changes or drops them. Also return:
- search_text: one standalone search sentence covering everything the user
  wants now, written as if typed fresh. If the user points at a shown job
  ("like the second one"), describe that job's role in it.
- require_salary: true only if the user wants jobs that state their pay."""


def schema(parse_schema: dict) -> dict:
    out = json.loads(json.dumps(parse_schema))
    out["properties"]["search_text"] = {"type": "string"}
    out["properties"]["require_salary"] = {"type": "boolean"}
    out["required"] = [*out["required"], "search_text", "require_salary"]
    return out


def describe_turn(n: int, turn: dict, titles: list[str]) -> str:
    shown = "\n".join(f"    {i}. {t}" for i, t in enumerate(titles, 1)) or "    (none)"
    return (f"Turn {n}\n  User: {turn['q']}\n  Filters after it: {json.dumps(turn['filters'])}\n"
            f"  Jobs shown:\n{shown}")


def request(history: list[dict], titles: list[list[str]], message: str, output_schema: dict) -> dict:
    turns = "\n\n".join(describe_turn(n, t, ts) for n, (t, ts) in enumerate(zip(history, titles), 1))
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM,
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": output_schema}},
        "messages": [{"role": "user", "content": f"{turns}\n\nNew message: {message}"}],
    }


def clean_filters(filters: dict, parse_schema: dict) -> dict:
    """Keep only allowed values; history comes back from the browser."""
    props = parse_schema["properties"]
    out = {}
    for field in ("seniority", "categories", "countries", "employment_types"):
        allowed = set(props[field]["items"]["enum"])
        values = filters.get(field)
        # Room for a region spelled out as countries ("anywhere in Europe").
        out[field] = [v for v in values if v in allowed][:80] if isinstance(values, list) else []
    salary = filters.get("salary_min_usd")
    out["salary_min_usd"] = salary if isinstance(salary, int) and 0 <= salary <= 1_000_000 else 0
    return out
