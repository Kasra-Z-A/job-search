import datetime as dt
import json

import numpy as np
import pytest

from job_search.app import bundle, store
from job_search.app.handler import App
from job_search.refresh import check, crawl_new, in_window, refresh

NOW = dt.datetime(2026, 10, 20, 6, 0, tzinfo=dt.UTC)
DIM = 4


def ts(days_ago: float) -> int:
    return int((NOW - dt.timedelta(days=days_ago)).timestamp())


def posting(guid: str, days_ago: float, **overrides) -> dict:
    base = {"guid": guid, "title": f"Engineer {guid}", "companyName": "Acme", "pubDate": ts(days_ago),
            "expiryDate": ts(days_ago - 60), "employmentType": "Full Time", "seniority": ["Senior"],
            "locationRestrictions": ["Canada"], "parentCategories": ["Developer"], "categories": [],
            "minSalary": 100_000, "maxSalary": 120_000, "currency": "USD", "salaryPeriod": "annual",
            "description": f"<p>Build things at {guid}.</p>"}
    return {**base, **overrides}


class Feed:
    """Pages of postings, newest first, like the Himalayas API."""

    def __init__(self, postings, page_size=2):
        self.pages = [postings[i:i + page_size] for i in range(0, len(postings), page_size)]
        self.requests = 0

    def __call__(self, cursor):
        self.requests += 1
        i = int(cursor or 0)
        return {"jobs": self.pages[i], "nextCursor": str(i + 1) if i + 1 < len(self.pages) else None}


def crawl(feed, known=(), **kw):
    return crawl_new(feed, set(known), NOW - dt.timedelta(days=7), sleep=lambda s: None, **kw)


def test_crawl_stops_at_a_page_it_already_knows():
    feed = Feed([posting(g, 0.1 * i) for i, g in enumerate("abcdef")])
    jobs, info = crawl(feed, known={"c", "d", "e", "f"})
    assert [j["guid"] for j in jobs] == ["a", "b"]
    assert info == {"pages": 2, "stopped_because": "caught_up"}


def test_crawl_skips_known_jobs_on_a_mixed_page_and_keeps_going():
    feed = Feed([posting(g, 0.1 * i) for i, g in enumerate("abcd")])
    jobs, info = crawl(feed, known={"b"})
    assert [j["guid"] for j in jobs] == ["a", "c", "d"]
    assert info["stopped_because"] == "end_of_feed"


def test_crawl_stops_once_postings_are_older_than_the_window():
    feed = Feed([posting("a", 1), posting("b", 6.9), posting("c", 7.5), posting("d", 8), posting("e", 9)])
    jobs, info = crawl(feed)
    assert [j["guid"] for j in jobs] == ["a", "b"]
    assert info == {"pages": 2, "stopped_because": "window"}


def test_crawl_respects_page_and_time_limits():
    postings = [posting(str(i), 0.01 * i) for i in range(20)]
    assert crawl(Feed(postings), max_pages=3)[1] == {"pages": 3, "stopped_because": "max_pages"}
    ticks = iter(range(100))
    jobs, info = crawl(Feed(postings), seconds=2.5, clock=lambda: next(ticks))
    assert info == {"pages": 2, "stopped_because": "time"}


def test_in_window_drops_old_and_expired_jobs():
    cutoff = NOW - dt.timedelta(days=7)
    iso = lambda days_ago: (NOW - dt.timedelta(days=days_ago)).isoformat()  # noqa: E731
    assert in_window({"published_at": iso(1), "expires_at": iso(-50)}, NOW, cutoff)
    assert in_window({"published_at": iso(1), "expires_at": None}, NOW, cutoff)
    assert not in_window({"published_at": iso(8), "expires_at": iso(-50)}, NOW, cutoff)
    assert not in_window({"published_at": iso(1), "expires_at": iso(0.5)}, NOW, cutoff)


def test_check_refuses_small_or_shrunken_bundles():
    check(1200, 0)
    check(1200, 2000)
    with pytest.raises(ValueError):
        check(900, 0)
    with pytest.raises(ValueError):
        check(1200, 3000)


def fake_embed(texts):
    vectors = np.ones((len(texts), DIM), dtype=np.float32)
    return vectors, 10 * len(texts)


def english(texts):
    return ["de" if "Bauen" in t else "en" for t in texts]


def run(target, feed, now=NOW, tmp=None, **kw):
    return refresh(target, feed, fake_embed, lambda day: ("2026-10-19", {"EUR": 1.0, "USD": 1.1}),
                   now=now, work_dir=tmp, min_jobs=1, detect=english, **kw)


def test_refresh_builds_publishes_and_then_adds_only_new_jobs(tmp_path):
    target = store.DirStore(tmp_path / "store")
    first = Feed([posting("a", 1), posting("b", 2, description="<p>Bauen Sie</p>"), posting("c", 3),
                  posting("d", 10), posting("x", 3, seniority=["Wizard"])])
    summary = run(target, first, tmp=tmp_path / "w1")
    assert summary["jobs"] == 2 and summary["not_english"] == 1 and summary["unreadable"] == 1
    assert summary["embedding_tokens"] == 20
    pointer = store.current(target)
    assert pointer["version"] == "20261020T060000Z" and pointer["jobs"] == 2

    later = NOW + dt.timedelta(hours=12)
    second = Feed([posting("n", -0.4), posting("a", 1), posting("c", 3)], page_size=1)
    summary = run(target, second, now=later, tmp=tmp_path / "w2")
    assert summary["added"] == 1 and summary["stopped_because"] == "caught_up"
    assert summary["embedding_tokens"] == 10  # only the new job

    jobs, texts, vectors, values = bundle.read(tmp_path / "store" / store.PREFIX / store.current(target)["version"])
    assert [j["id"] for j in jobs] == ["n", "a", "c"]  # newest first
    assert texts[0] == "Build things at n." and vectors.shape == (3, DIM)
    assert values["jobs"] == 3 and values["fx_date"] == "2026-10-19" and values["window_days"] == 7
    assert values["countries"] == ["Canada"]


def test_refresh_drops_jobs_that_left_the_window(tmp_path):
    target = store.DirStore(tmp_path / "store")
    run(target, Feed([posting("a", 1), posting("b", 6.5)]), tmp=tmp_path / "w1")
    summary = run(target, Feed([posting("a", 1)]), now=NOW + dt.timedelta(days=1), tmp=tmp_path / "w2")
    assert summary["jobs"] == 1 and summary["dropped"] == 1


def test_failed_check_keeps_the_old_bundle_live(tmp_path):
    target = store.DirStore(tmp_path / "store")
    run(target, Feed([posting(g, 1) for g in "abcd"]), tmp=tmp_path / "w1")
    before = store.current(target)
    # A day later only two of the four jobs are left: below half, so refused.
    with pytest.raises(ValueError):
        refresh(target, Feed([posting("a", 1)]), fake_embed, lambda d: ("d", {"EUR": 1.0, "USD": 1.1}),
                now=NOW + dt.timedelta(days=6.5), work_dir=tmp_path / "w2", min_jobs=1, detect=english)
    assert store.current(target) == before


def test_prune_keeps_the_newest_versions(tmp_path):
    target = store.DirStore(tmp_path)
    for version in ["v1", "v2", "v3", "v4"]:
        target.write(f"{store.PREFIX}{version}/values.json", b"{}")
    target.write(store.POINTER, json.dumps({"version": "v4"}).encode())
    assert store.prune(target, keep=2) == ["v1", "v2"]
    assert target.versions() == ["v3", "v4"]


def test_app_switches_bundles_when_the_pointer_moves(tmp_path):
    target = store.DirStore(tmp_path / "store")
    for version in ["v1", "v2"]:
        for name in store.FILES:
            target.write(f"{store.PREFIX}{version}/{name}", version.encode())
    target.write(store.POINTER, json.dumps({"version": "v1"}).encode())
    now = [0.0]
    app = App(lambda path, version: path.name, counter=None, target=target, cache=tmp_path / "cache",
              clock=lambda: now[0])
    assert app.current() == "v1"

    target.write(store.POINTER, json.dumps({"version": "v2"}).encode())
    now[0] = 100
    assert app.current() == "v1"  # pointer not reread yet
    now[0] = 400
    assert app.current() == "v2"
    assert not (tmp_path / "cache" / "v1").exists()
