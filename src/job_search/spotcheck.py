"""Hand spot-check of Himalayas seniority tags before they are used as labels.

`sample` writes a CSV with a fixed-seed sample of jobs per seniority level and
the evidence a reader needs: title, any "N years" mention and the start of the
text. The reader fills in `verdict` (ok, wrong or unclear) and, for wrong
ones, `should_be`. `summary` reports agreement per level, and `render` writes a readable
Markdown view next to the CSV.

    uv run python -m job_search.spotcheck sample data/normalised/himalayas-2026-10-13.jsonl.gz
    uv run python -m job_search.spotcheck summary data/review/seniority-spotcheck-2026-10-13.csv
    uv run python -m job_search.spotcheck render data/review/seniority-spotcheck-2026-10-13.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import random
import re
from pathlib import Path

from job_search.normalise import SENIORITY_LEVELS, snapshot_date

PER_LEVEL = 8
SEED = 20261013
REVIEW_DIR = Path("data/review")
FIELDS = ["n", "tag", "title", "company", "years_mentioned", "text_start", "url", "verdict", "should_be", "note"]
VERDICTS = {"ok", "wrong", "unclear"}
YEARS = re.compile(r"[^.\n]{0,60}\b\d+\s*\+?\s*(?:[-–]|to)?\s*\d*\s*\+?\s*years?\b[^.\n]{0,60}", re.I)


def years_mentioned(text: str) -> str:
    return " | ".join(m.group(0).strip() for m in YEARS.finditer(text))[:300]


def sample_jobs(jobs: list[dict], per_level: int = PER_LEVEL, seed: int = SEED) -> list[dict]:
    """`per_level` jobs tagged with each level; a job is used at most once."""
    rng = random.Random(seed)
    used: set[str] = set()
    rows = []
    for level in SENIORITY_LEVELS:
        pool = [j for j in jobs if level in j["seniority"] and j["id"] not in used]
        for job in rng.sample(pool, min(per_level, len(pool))):
            used.add(job["id"])
            rows.append({
                "n": len(rows) + 1,
                "tag": ", ".join(job["seniority"]),
                "title": job["title"],
                "company": job["company"],
                "years_mentioned": years_mentioned(job["text"]),
                "text_start": job["text"][:400].replace("\n", " / "),
                "url": job["url"],
                "verdict": "",
                "should_be": "",
                "note": "",
            })
    return rows


def write_sample(normalised: Path, out_dir: Path = REVIEW_DIR) -> Path:
    with gzip.open(normalised, "rt", encoding="utf-8") as f:
        jobs = [json.loads(line) for line in f]
    out = out_dir / f"seniority-spotcheck-{snapshot_date(normalised)}.csv"
    if out.exists():
        raise FileExistsError(f"{out} exists; it may hold verdicts already")
    out_dir.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sample_jobs(jobs))
    return out


def summarise(rows: list[dict]) -> dict:
    by_level: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    pending = 0
    for row in rows:
        verdict = row["verdict"].strip().lower()
        if not verdict:
            pending += 1
            continue
        if verdict not in VERDICTS:
            raise ValueError(f"row {row['n']}: verdict must be one of {sorted(VERDICTS)}, got {verdict!r}")
        by_level[row["tag"]][verdict] += 1

    total = sum(by_level.values(), collections.Counter())
    judged = total["ok"] + total["wrong"]
    return {
        "judged": sum(total.values()),
        "pending": pending,
        "agreement": round(total["ok"] / judged, 3) if judged else None,
        "by_tag": {tag: dict(c) for tag, c in sorted(by_level.items())},
        "corrections": collections.Counter(
            f"{r['tag']} -> {r['should_be'].strip()}" for r in rows
            if r["verdict"].strip().lower() == "wrong" and r["should_be"].strip()
        ).most_common(),
    }


VERDICT_ORDER = {"wrong": 0, "unclear": 1, "": 2, "ok": 3}


def render_markdown(rows: list[dict], title: str) -> str:
    """A readable view of the CSV: wrong and unclear first, one section per job."""
    rows = sorted(rows, key=lambda r: (VERDICT_ORDER.get(r["verdict"].strip().lower(), 2), int(r["n"])))
    out = [f"# {title}", "", "Generated from the CSV, which holds the verdicts; edit that, not this.", ""]
    for r in rows:
        verdict = r["verdict"].strip().lower() or "pending"
        heading = f"## #{r['n']} · {verdict}"
        if r.get("should_be"):
            heading += f" → should be {r['should_be']}"
        out += [heading, "", f"**{r['title']}** at {r['company']}  ", f"Tag: **{r['tag']}**  ", f"[Posting]({r['url']})", ""]
        if r.get("note"):
            out += [f"> {r['note']}", ""]
        if r["years_mentioned"]:
            out += [f"*Years mentioned:* {r['years_mentioned']}", ""]
        out += [r["text_start"].replace(" / ", "\n\n"), ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sample").add_argument("normalised", type=Path)
    sub.add_parser("summary").add_argument("csv", type=Path)
    sub.add_parser("render").add_argument("csv", type=Path)
    args = parser.parse_args(argv)

    if args.command == "sample":
        print(write_sample(args.normalised))
        return
    with args.csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.command == "summary":
        print(json.dumps(summarise(rows), indent=2))
    else:
        out = args.csv.with_suffix(".md")
        out.write_text(render_markdown(rows, f"Seniority spot-check: {args.csv.stem}"), encoding="utf-8")
        print(out)


if __name__ == "__main__":
    main()
