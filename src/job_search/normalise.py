"""Turn a raw snapshot into clean, comparable job records.

Every later step (query generation, answer sets, indexing) reads the output
of this module, never the raw API fields. The rules:

- Salary is converted to yearly USD with the ECB reference rates of the
  snapshot date (stored in `data/fx/`, committed) and a full-time year:
  hourly x 2080, weekly x 52, monthly x 12. A yearly figure outside
  [MIN_PLAUSIBLE_USD, MAX_PLAUSIBLE_USD] is treated as unknown; such values
  are almost always monthly pay labelled as annual.
- `salary_status` says why a salary is or is not usable: stated, none,
  unconvertible (currency without an ECB rate) or implausible.
- Seniority keeps Himalayas' levels, deduplicated and in level order.
- An empty country list means the job is open worldwide.
- Categories are Himalayas' top-level categories; an empty list means unknown,
  not "no category".
- The HTML description becomes plain text with one block per line.

    uv run python -m job_search.normalise data/snapshots/himalayas-2026-10-13.jsonl.gz
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import gzip
import hashlib
import html
import io
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import httpx

SENIORITY_LEVELS = ["Entry-level", "Mid-level", "Senior", "Manager", "Director", "Executive"]
HOURS_PER_YEAR = 2080
PERIOD_FACTOR = {"annual": 1, "monthly": 12, "weekly": 52, "hourly": HOURS_PER_YEAR}
MIN_PLAUSIBLE_USD = 5_000
MAX_PLAUSIBLE_USD = 1_000_000

FX_DIR = Path("data/fx")
OUT_DIR = Path("data/normalised")
ECB_URL = "https://data-api.ecb.europa.eu/service/data/EXR/D..EUR.SP00.A"


# --- exchange rates -------------------------------------------------------


def fx_path(date: str, fx_dir: Path = FX_DIR) -> Path:
    return fx_dir / f"ecb-{date}.json"


def fetch_ecb_rates(date: str, client: httpx.Client) -> dict[str, float]:
    """ECB reference rates for `date`, as units of currency per 1 EUR."""
    response = client.get(
        ECB_URL, params={"startPeriod": date, "endPeriod": date, "format": "csvdata"}
    )
    response.raise_for_status()
    rows = csv.DictReader(io.StringIO(response.text))
    rates = {row["CURRENCY"]: float(row["OBS_VALUE"]) for row in rows if row["OBS_VALUE"]}
    if "USD" not in rates:
        raise ValueError(f"no ECB rates for {date} (weekend or holiday?)")
    return {"EUR": 1.0, **dict(sorted(rates.items()))}


def load_or_fetch_rates(date: str, fx_dir: Path = FX_DIR) -> dict[str, float]:
    path = fx_path(date, fx_dir)
    if path.exists():
        return json.loads(path.read_text())["per_eur"]
    with httpx.Client(timeout=30) as client:
        rates = fetch_ecb_rates(date, client)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"source": "ECB euro foreign exchange reference rates", "date": date, "per_eur": rates}
    path.write_text(json.dumps(data, indent=2) + "\n")
    return rates


# --- field rules ----------------------------------------------------------


def normalise_salary(job: dict, per_eur: dict[str, float]) -> dict:
    lo, hi = job.get("minSalary"), job.get("maxSalary")
    if not lo and not hi:
        return {"salary_status": "none", "salary_min_usd": None, "salary_max_usd": None}
    lo, hi = lo or hi, hi or lo
    currency, period = job.get("currency"), job.get("salaryPeriod")
    if currency not in per_eur or period not in PERIOD_FACTOR:
        return {"salary_status": "unconvertible", "salary_min_usd": None, "salary_max_usd": None}

    to_usd = per_eur["USD"] / per_eur[currency] * PERIOD_FACTOR[period]
    lo_usd, hi_usd = round(lo * to_usd), round(hi * to_usd)
    if lo_usd < MIN_PLAUSIBLE_USD or hi_usd > MAX_PLAUSIBLE_USD:
        return {"salary_status": "implausible", "salary_min_usd": None, "salary_max_usd": None}
    return {"salary_status": "stated", "salary_min_usd": lo_usd, "salary_max_usd": hi_usd}


def normalise_seniority(levels: list[str] | None) -> list[str]:
    given = set(levels or [])
    unknown = given - set(SENIORITY_LEVELS)
    if unknown:
        raise ValueError(f"unknown seniority levels: {sorted(unknown)}")
    return [level for level in SENIORITY_LEVELS if level in given]


class _TextExtractor(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "ul", "ol", "tr", "table", "section",
             "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "hr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.BLOCK:
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("- ")

    def handle_endtag(self, tag):
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        # Newlines in the HTML source are not line breaks; only block tags are.
        self.parts.append(re.sub(r"\s+", " ", data))


def html_to_text(markup: str | None) -> str:
    parser = _TextExtractor()
    parser.feed(markup or "")
    parser.close()
    lines = (re.sub(r"\s+", " ", html.unescape(line)).strip() for line in "".join(parser.parts).split("\n"))
    return "\n".join(line for line in lines if line and line != "-")


def _iso(timestamp: int | None) -> str | None:
    if timestamp is None:
        return None
    return dt.datetime.fromtimestamp(timestamp, dt.UTC).isoformat(timespec="seconds")


def normalise_job(job: dict, per_eur: dict[str, float]) -> dict:
    countries = sorted(set(job.get("locationRestrictions") or []))
    return {
        "id": job["guid"],
        "url": job["guid"],
        "title": job["title"].strip(),
        "company": job.get("companyName"),
        "published_at": _iso(job.get("pubDate")),
        "expires_at": _iso(job.get("expiryDate")),
        "employment_type": job.get("employmentType"),
        "seniority": normalise_seniority(job.get("seniority")),
        "countries": countries,
        "worldwide": not countries,
        "timezones": sorted(set(job.get("timezoneRestrictions") or [])),
        "categories": sorted(set(job.get("parentCategories") or [])),
        "subcategories": sorted(set(job.get("categories") or [])),
        **normalise_salary(job, per_eur),
        "salary_currency": job.get("currency"),
        "salary_period": job.get("salaryPeriod"),
        "salary_min_raw": job.get("minSalary"),
        "salary_max_raw": job.get("maxSalary"),
        "text": html_to_text(job.get("description")),
    }


# --- run ------------------------------------------------------------------


def snapshot_date(snapshot: Path) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", snapshot.name)
    if not match:
        raise ValueError(f"no date in snapshot name {snapshot.name}")
    return match.group(1)


def verify_snapshot(snapshot: Path) -> None:
    manifest = snapshot.with_name(snapshot.name.replace(".jsonl.gz", ".manifest.json"))
    expected = json.loads(manifest.read_text())["sha256"]
    actual = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"{snapshot} does not match its manifest sha256")


def normalise_snapshot(snapshot: Path, out_dir: Path = OUT_DIR, fx_dir: Path = FX_DIR) -> dict:
    verify_snapshot(snapshot)
    date = snapshot_date(snapshot)
    per_eur = load_or_fetch_rates(date, fx_dir)

    with gzip.open(snapshot, "rt", encoding="utf-8") as f:
        jobs = [normalise_job(json.loads(line), per_eur) for line in f]

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / snapshot.name
    with out.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        for job in jobs:
            gz.write((json.dumps(job, ensure_ascii=False) + "\n").encode())

    return {"output": str(out), **summarise(jobs)}


def summarise(jobs: list[dict]) -> dict:
    count = collections.Counter
    return {
        "jobs": len(jobs),
        "salary_status": dict(count(j["salary_status"] for j in jobs).most_common()),
        "seniority": dict(count(s for j in jobs for s in j["seniority"]).most_common()),
        "employment_type": dict(count(j["employment_type"] for j in jobs).most_common()),
        "worldwide": sum(j["worldwide"] for j in jobs),
        "category_unknown": sum(not j["categories"] for j in jobs),
        "empty_text": sum(not j["text"] for j in jobs),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    print(json.dumps(normalise_snapshot(args.snapshot, args.out_dir), indent=2))


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        sys.exit(f"error: {exc}")
