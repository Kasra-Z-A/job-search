"""Keep the live demo's jobs current: a rolling window of recent postings.

The evaluation runs on frozen snapshots and never sees these jobs; the live
site runs the same pipeline (normalise, embed, bundle) on whatever Himalayas
published in the last WINDOW_DAYS. Each run:

1. reads the current bundle from the store (`job_search.app.store`);
2. crawls the feed, newest first, until a whole page is already known, the
   postings are older than the window, or it runs out of pages or time;
3. normalises the new postings with the latest ECB rates and keeps English
   ones, as `job_search.normalise` does for snapshots;
4. embeds only the new jobs (~835 tokens each, $0.02 per million tokens);
5. drops jobs published before the window or past their expiry date;
6. checks the result, writes it as a new version and moves the pointer;
   older versions beyond KEEP_VERSIONS are deleted.

A run that stops early (pages or time) leaves a gap in the window that ages
out; it never serves a broken bundle. The OpenAI key comes from
OPENAI_API_KEY, or from the SSM parameter named in OPENAI_KEY_PARAM on Lambda.

    uv run python -m job_search.refresh --bucket jobs-kasra-page-artifacts-<account>
    uv run python -m job_search.refresh --dir data/live      # local folder instead of S3
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import httpx
import numpy as np

from job_search.app import bundle, store
from job_search.normalise import (
    KEEP_LANGUAGE,
    LANGUAGE_SAMPLE_CHARS,
    detect_languages,
    fetch_latest_ecb_rates,
    normalise_job,
)
from job_search.snapshot import USER_AGENT, GetPage, http_get_page
from job_search.systems import job_text
from job_search.systems.embeddings import MAX_CHARS, PRICE_PER_MILLION, embed

WINDOW_DAYS = 7
MAX_PAGES = 500  # 10,000 postings, about two days of the feed
CRAWL_SECONDS = 600  # Lambda stops a run at 15 minutes; leave time for the rest
KEEP_VERSIONS = 4
MIN_JOBS = 1000
MIN_SHARE_OF_PREVIOUS = 0.5


def crawl_new(
    get_page: GetPage,
    known: set[str],
    cutoff: dt.datetime,
    *,
    max_pages: int = MAX_PAGES,
    seconds: float = CRAWL_SECONDS,
    delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[list[dict], dict]:
    """Raw postings newer than cutoff and not yet known, newest first, and why the crawl stopped."""
    deadline = clock() + seconds
    cutoff_ts = cutoff.timestamp()
    jobs, seen, cursor, pages = [], set(), None, 0
    while True:
        if pages >= max_pages:
            stopped = "max_pages"
            break
        if clock() >= deadline:
            stopped = "time"
            break
        if pages:
            sleep(delay)
        page = get_page(cursor)
        pages += 1
        batch = [j for j in page.get("jobs") or [] if j.get("guid")]
        for job in batch:
            guid = job["guid"]
            if guid in known or guid in seen or (job.get("pubDate") or 0) < cutoff_ts:
                continue
            seen.add(guid)
            jobs.append(job)
        cursor = page.get("nextCursor")
        if not batch or not cursor:
            stopped = "end_of_feed"
            break
        # The feed is ordered by publication time, so nothing further back is new.
        if all(j["guid"] in known for j in batch):
            stopped = "caught_up"
            break
        if max(j.get("pubDate") or 0 for j in batch) < cutoff_ts:
            stopped = "window"
            break
        if pages % 25 == 0:
            print(f"  {pages} pages, {len(jobs)} new", file=sys.stderr)
    return jobs, {"pages": pages, "stopped_because": stopped}


def normalise_new(raw: list[dict], per_eur: dict[str, float], detect=detect_languages) -> tuple[list[dict], dict]:
    """English, normalised jobs; postings the rules cannot read are skipped and counted."""
    jobs, failed = [], 0
    for posting in raw:
        try:
            jobs.append(normalise_job(posting, per_eur))
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            failed += 1
            print(f"  skipped {posting.get('guid')}: {type(exc).__name__}: {exc}", file=sys.stderr)
    languages = detect([j["text"][:LANGUAGE_SAMPLE_CHARS] for j in jobs]) if jobs else []
    english = [j for j, lang in zip(jobs, languages) if lang == KEEP_LANGUAGE]
    return english, {"unreadable": failed, "not_english": len(jobs) - len(english)}


def in_window(job: dict, now: dt.datetime, cutoff: dt.datetime) -> bool:
    published = job.get("published_at")
    expires = job.get("expires_at")
    if not published or dt.datetime.fromisoformat(published) < cutoff:
        return False
    return not expires or dt.datetime.fromisoformat(expires) > now


def check(count: int, previous: int, min_jobs: int = MIN_JOBS) -> None:
    """Refuse a bundle that is suspiciously small, so a bad run keeps the old one live."""
    if count < min_jobs:
        raise ValueError(f"only {count} jobs, fewer than {min_jobs}")
    if previous and count < MIN_SHARE_OF_PREVIOUS * previous:
        raise ValueError(f"only {count} jobs against {previous} before")


def refresh(
    target,
    get_page: GetPage,
    embed_texts: Callable[[list[str]], tuple[np.ndarray, int]],
    rates: Callable[[str], tuple[str, dict[str, float]]],
    *,
    now: dt.datetime,
    work_dir: Path,
    window_days: int = WINDOW_DAYS,
    max_pages: int = MAX_PAGES,
    seconds: float = CRAWL_SECONDS,
    min_jobs: int = MIN_JOBS,
    keep_versions: int = KEEP_VERSIONS,
    detect=detect_languages,
) -> dict:
    cutoff = now - dt.timedelta(days=window_days)
    pointer = store.current(target)
    if pointer:
        jobs, texts, vectors, _ = bundle.read(store.fetch(target, pointer["version"], work_dir))
    else:
        jobs, texts, vectors = [], [], np.zeros((0, 0), dtype=np.float16)

    raw, crawled = crawl_new(get_page, {j["id"] for j in jobs}, cutoff, max_pages=max_pages, seconds=seconds)
    rates_day, per_eur = rates(now.date().isoformat())
    new, skipped = normalise_new(raw, per_eur, detect)
    new_vectors, tokens = embed_texts([job_text(j)[:MAX_CHARS] for j in new]) if new else (None, 0)

    rows = [(bundle.slim(j), j["text"], v) for j, v in zip(new, new_vectors if new else [])]
    rows += list(zip(jobs, texts, vectors))
    kept = sorted((r for r in rows if in_window(r[0], now, cutoff)),
                  key=lambda r: r[0]["published_at"], reverse=True)
    check(len(kept), len(jobs), min_jobs)

    version = now.strftime("%Y%m%dT%H%M%SZ")
    updated_at = now.isoformat(timespec="seconds")
    out = bundle.write(work_dir / f"{version}.new", [r[0] for r in kept], [r[1] for r in kept],
                       np.stack([r[2] for r in kept]),
                       {"snapshot": now.date().isoformat(), "updated_at": updated_at,
                        "window_days": window_days, "fx_date": rates_day})
    store.publish(target, out, version, {"jobs": len(kept), "updated_at": updated_at})
    deleted = store.prune(target, keep_versions)

    return {"version": version, "jobs": len(kept), "previous_jobs": len(jobs), "new_postings": len(raw),
            "added": len(new), "dropped": len(jobs) + len(new) - len(kept), **skipped, **crawled,
            "fx_date": rates_day, "embedding_tokens": tokens,
            "embedding_cost_usd": round(tokens * PRICE_PER_MILLION / 1e6, 4), "deleted_versions": deleted}


# --- entry points ---------------------------------------------------------


def _openai_key() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    import boto3

    param = os.environ["OPENAI_KEY_PARAM"]
    return boto3.client("ssm").get_parameter(Name=param, WithDecryption=True)["Parameter"]["Value"]


def run(target, **options) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=_openai_key(), max_retries=5, timeout=60)
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30) as http, \
            tempfile.TemporaryDirectory() as work:
        return refresh(target, http_get_page(http), lambda texts: embed(client, texts),
                       lambda day: fetch_latest_ecb_rates(day, http),
                       now=dt.datetime.now(dt.UTC).replace(microsecond=0), work_dir=Path(work), **options)


def lambda_handler(event, context):
    import boto3

    summary = run(store.S3Store(boto3.client("s3"), os.environ["BUNDLE_BUCKET"]))
    print(json.dumps(summary))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--bucket", help="S3 bucket holding bundles/")
    where.add_argument("--dir", type=Path, help="local folder instead of S3")
    parser.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES)
    parser.add_argument("--crawl-minutes", type=float, default=CRAWL_SECONDS / 60)
    parser.add_argument("--min-jobs", type=int, default=MIN_JOBS)
    args = parser.parse_args(argv)

    if args.bucket:
        import boto3

        target = store.S3Store(boto3.client("s3"), args.bucket)
    else:
        target = store.DirStore(args.dir)
    summary = run(target, window_days=args.window_days, max_pages=args.max_pages,
                  seconds=args.crawl_minutes * 60, min_jobs=args.min_jobs)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
