"""Download a frozen, dated snapshot of Himalayas job postings.

Evaluation answer sets are computed from a snapshot, so a snapshot never
changes once written. A crawl appends to a partial file and records its cursor
after every page; an interrupted crawl resumes from there. Only a finished
crawl produces the final files:

    data/snapshots/himalayas-<date>.jsonl.gz       one raw API job per line
    data/snapshots/himalayas-<date>.manifest.json  counts, timing, sha256

Jobs are stored exactly as the API returns them; normalisation is a separate step.

    uv run python -m job_search.snapshot --limit 10000
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

API_URL = "https://himalayas.app/jobs/api"
PAGE_SIZE = 20  # the API's maximum per request
USER_AGENT = "job-search snapshot (github.com/Kasra-Z-A/job-search)"
DEFAULT_DIR = Path("data/snapshots")

GetPage = Callable[[str | None], dict]


@dataclass(frozen=True)
class SnapshotPaths:
    final: Path
    manifest: Path
    partial: Path
    state: Path

    @classmethod
    def for_date(cls, out_dir: Path, date: str) -> SnapshotPaths:
        stem = f"himalayas-{date}"
        return cls(
            final=out_dir / f"{stem}.jsonl.gz",
            manifest=out_dir / f"{stem}.manifest.json",
            partial=out_dir / f".{stem}.partial.jsonl",
            state=out_dir / f".{stem}.state.json",
        )


def http_get_page(
    client: httpx.Client,
    *,
    max_attempts: int = 6,
    backoff: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> GetPage:
    """Return a page fetcher that retries rate limits, server errors and timeouts."""

    def get_page(cursor: str | None) -> dict:
        params = {"limit": PAGE_SIZE}
        if cursor:
            params["cursor"] = cursor
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.get(API_URL, params=params)
            except httpx.TransportError as exc:
                error = f"{type(exc).__name__}: {exc}"
                wait = backoff**attempt
            else:
                if response.status_code == 200:
                    return response.json()
                if response.status_code != 429 and response.status_code < 500:
                    response.raise_for_status()
                error = f"HTTP {response.status_code}"
                wait = _retry_after(response) or backoff**attempt
            if attempt == max_attempts:
                raise RuntimeError(f"giving up after {max_attempts} attempts: {error}")
            print(f"  {error}, retrying in {wait:.0f}s", file=sys.stderr)
            sleep(wait)
        raise AssertionError("unreachable")

    return get_page


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers["Retry-After"])
    except (KeyError, ValueError):
        return None


def crawl(
    get_page: GetPage,
    paths: SnapshotPaths,
    *,
    limit: int,
    delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
) -> dict:
    """Crawl until `limit` unique jobs or the end of the feed; return the manifest."""
    if paths.final.exists():
        raise FileExistsError(f"{paths.final} exists; snapshots are never overwritten")
    paths.partial.parent.mkdir(parents=True, exist_ok=True)

    state = _load_state(paths)
    seen = _load_seen(paths.partial)
    if seen:
        print(f"resuming with {len(seen)} jobs", file=sys.stderr)
    # Drop lines written after the last recorded cursor (a crash between the
    # two writes); the pages they came from are fetched again.
    if len(seen) != state["count"]:
        seen = _truncate_partial(paths.partial, state["count"])
        state["count"] = len(seen)

    with paths.partial.open("a", encoding="utf-8") as out:
        while len(seen) < limit and not state["finished_feed"]:
            if state["pages"]:
                sleep(delay)
            page = get_page(state["cursor"])
            jobs = page.get("jobs") or []
            if state["pages"] == 0:
                state["api_total_count"] = page.get("totalCount")
                state["api_updated_at"] = page.get("updatedAt")
            state["pages"] += 1

            for job in jobs:
                if len(seen) >= limit:
                    break
                guid = job.get("guid")
                if not guid:
                    state["skipped_no_guid"] += 1
                    continue
                if guid in seen:
                    state["duplicates"] += 1
                    continue
                seen.add(guid)
                out.write(json.dumps(job, ensure_ascii=False) + "\n")
            out.flush()

            state["count"] = len(seen)
            state["cursor"] = page.get("nextCursor")
            state["finished_feed"] = not jobs or not state["cursor"]
            _write_json(paths.state, state)
            if state["pages"] % 25 == 0:
                print(f"  {state['pages']} pages, {len(seen)} jobs", file=sys.stderr)

    return _finish(paths, state, limit, now())


def _load_state(paths: SnapshotPaths) -> dict:
    if paths.state.exists():
        return json.loads(paths.state.read_text())
    return {
        "started_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "cursor": None,
        "pages": 0,
        "count": 0,
        "duplicates": 0,
        "skipped_no_guid": 0,
        "finished_feed": False,
        "api_total_count": None,
        "api_updated_at": None,
    }


def _load_seen(partial: Path) -> set[str]:
    if not partial.exists():
        return set()
    with partial.open(encoding="utf-8") as f:
        return {json.loads(line)["guid"] for line in f if line.strip()}


def _truncate_partial(partial: Path, keep: int) -> set[str]:
    lines = partial.read_text(encoding="utf-8").splitlines(keepends=True)[:keep]
    partial.write_text("".join(lines), encoding="utf-8")
    return {json.loads(line)["guid"] for line in lines}


def _finish(paths: SnapshotPaths, state: dict, limit: int, finished_at: dt.datetime) -> dict:
    # mtime=0 and no filename make the gzip bytes depend only on the content.
    with paths.partial.open("rb") as src, paths.final.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            shutil.copyfileobj(src, gz)

    manifest = {
        "source": "Himalayas",
        "api_url": API_URL,
        "snapshot": paths.final.name,
        "jobs": state["count"],
        "limit": limit,
        "stopped_because": "end_of_feed" if state["finished_feed"] else "limit",
        "pages": state["pages"],
        "duplicates_skipped": state["duplicates"],
        "skipped_no_guid": state["skipped_no_guid"],
        "api_total_count": state["api_total_count"],
        "api_updated_at": state["api_updated_at"],
        "started_at": state["started_at"],
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "sha256": hashlib.sha256(paths.final.read_bytes()).hexdigest(),
        "attribution": "Job data from Himalayas (himalayas.app); link each job to its original posting.",
    }
    _write_json(paths.manifest, manifest)
    paths.partial.unlink()
    paths.state.unlink()
    return manifest


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=10_000, help="unique jobs to collect")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="snapshot date label")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args(argv)

    paths = SnapshotPaths.for_date(args.out_dir, args.date)
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30) as client:
        manifest = crawl(http_get_page(client), paths, limit=args.limit, delay=args.delay)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
