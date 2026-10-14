"""Human-written queries for a sample of evaluation specs.

The evaluation queries were written by an LLM, and the main system reads
them with an LLM, so LLM wording might be easier for it than people's. To
check, a person writes their own query for a sample of the same specs; the
two versions of each spec are then scored side by side.

`worksheet` writes a Markdown file with one section per spec and an empty
"Query:" line to fill in. `collect` reads the filled-in worksheet into a
query file in the same format as the generated one, so every system and the
harness can run on it unchanged.

    uv run python -m job_search.human worksheet data/eval/queries-2026-10-13.jsonl
    uv run python -m job_search.human collect data/eval/human-worksheet-2026-10-13.md
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

SAMPLE = 40
SEED = 20261015
SENIORITY = {
    "Entry-level": "entry level",
    "Mid-level": "mid level",
    "Senior": "senior level",
    "Manager": "manager (manages people)",
    "Leadership": "leadership (director, head of, VP or executive)",
}
EMPLOYMENT = {"Full Time": "full-time", "Part Time": "part-time", "Contractor": "contract", "Intern": "internship"}
LABELS = {
    "seniority": SENIORITY.get,
    "category": lambda v: f"{v} jobs",
    "country": lambda v: f"open to people living in {v}",
    "employment_type": EMPLOYMENT.get,
    "salary_min_usd": lambda v: f"at least ${v:,} a year",
}
SECTION = re.compile(r"^## (s\d{4})\n(?:.*\n)*?Query:[ \t]*(.*)$", re.M)


def pick(queries: list[dict], n: int = SAMPLE, seed: int = SEED) -> list[dict]:
    """A fixed-seed sample with the same 2/3/4-constraint mix as the full set."""
    rng = random.Random(seed)
    by_size: dict[int, list[dict]] = {}
    for q in queries:
        by_size.setdefault(len(q["constraints"]), []).append(q)
    picked = []
    for size, group in sorted(by_size.items()):
        take = round(n * len(group) / len(queries))
        picked += rng.sample(group, take)
    return sorted(picked, key=lambda q: q["spec_id"])


def worksheet(queries: list[dict]) -> str:
    lines = [
        "# Write your own job search queries",
        "",
        "For each item, write the search you would type into a job board to find",
        "exactly these jobs. Use your own words, the way you would really search:",
        "short or long, casual or formal. Include every point and nothing else.",
        "All jobs are remote; salaries are yearly, in US dollars.",
        "",
        "Write on the `Query:` line, after the colon. Leave the rest as it is.",
        "",
    ]
    for q in queries:
        lines += [f"## {q['spec_id']}", ""]
        lines += [f"- {LABELS[field](value)}" for field, value in q["constraints"].items()]
        lines += ["", "Query: ", ""]
    return "\n".join(lines)


def collect(text: str, queries: list[dict]) -> list[dict]:
    """Specs with the human query in place of the generated one."""
    written = {sid: query.strip() for sid, query in SECTION.findall(text)}
    empty = [sid for sid, query in written.items() if not query]
    if empty:
        raise ValueError(f"no query written for {len(empty)} items, e.g. {empty[:3]}")
    by_id = {q["spec_id"]: q for q in queries}
    return [{**by_id[sid], "query": query, "style": "human"} for sid, query in written.items()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("worksheet").add_argument("queries", type=Path)
    sub.add_parser("collect").add_argument("worksheet", type=Path)
    args = parser.parse_args(argv)

    if args.command == "worksheet":
        queries = [json.loads(line) for line in args.queries.open()]
        out = args.queries.with_name(args.queries.name.replace("queries-", "human-worksheet-").replace(".jsonl", ".md"))
        if out.exists():
            raise SystemExit(f"{out} exists; it may hold written queries")
        out.write_text(worksheet(pick(queries)), encoding="utf-8")
        print(out)
    else:
        date = re.search(r"\d{4}-\d{2}-\d{2}", args.worksheet.name).group(0)
        source = args.worksheet.with_name(f"queries-{date}.jsonl")
        queries = [json.loads(line) for line in source.open()]
        rows = collect(args.worksheet.read_text(encoding="utf-8"), queries)
        out = args.worksheet.with_name(f"human-{date}.jsonl")
        out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        print(f"{out}: {len(rows)} queries")


if __name__ == "__main__":
    main()
