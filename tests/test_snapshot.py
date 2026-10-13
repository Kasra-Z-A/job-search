import gzip
import hashlib
import json

import httpx
import pytest

from job_search.snapshot import SnapshotPaths, crawl, http_get_page


def job(n):
    return {"guid": f"https://himalayas.app/jobs/{n}", "title": f"Job {n}"}


def fake_feed(pages):
    """pages: list of job lists; cursor i points at pages[i]."""
    calls = []

    def get_page(cursor):
        i = int(cursor or 0)
        calls.append(cursor)
        nxt = str(i + 1) if i + 1 < len(pages) else None
        return {"jobs": pages[i], "nextCursor": nxt, "totalCount": 99, "updatedAt": 123}

    return get_page, calls


def read_snapshot(paths):
    with gzip.open(paths.final, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture
def paths(tmp_path):
    return SnapshotPaths.for_date(tmp_path, "2026-10-13")


def no_sleep(_):
    pass


def test_crawls_to_end_of_feed_and_skips_duplicates(paths):
    get_page, _ = fake_feed([[job(1), job(2)], [job(2), job(3)], [{"title": "no guid"}]])

    manifest = crawl(get_page, paths, limit=100, sleep=no_sleep)

    assert [j["title"] for j in read_snapshot(paths)] == ["Job 1", "Job 2", "Job 3"]
    assert manifest["jobs"] == 3
    assert manifest["duplicates_skipped"] == 1
    assert manifest["skipped_no_guid"] == 1
    assert manifest["stopped_because"] == "end_of_feed"
    assert manifest["api_total_count"] == 99
    assert manifest["sha256"] == hashlib.sha256(paths.final.read_bytes()).hexdigest()
    assert not paths.partial.exists() and not paths.state.exists()


def test_stops_at_limit(paths):
    get_page, calls = fake_feed([[job(1), job(2)], [job(3), job(4)], [job(5)]])

    manifest = crawl(get_page, paths, limit=3, sleep=no_sleep)

    assert len(read_snapshot(paths)) == 3
    assert manifest["stopped_because"] == "limit"
    assert len(calls) == 2


def test_never_overwrites_a_snapshot(paths):
    get_page, _ = fake_feed([[job(1)]])
    crawl(get_page, paths, limit=10, sleep=no_sleep)

    with pytest.raises(FileExistsError):
        crawl(get_page, paths, limit=10, sleep=no_sleep)


def test_resumes_after_interruption(paths):
    pages = [[job(1), job(2)], [job(3), job(4)], [job(5)]]
    get_page, _ = fake_feed(pages)

    def failing(cursor):
        if cursor == "2":
            raise RuntimeError("network down")
        return get_page(cursor)

    with pytest.raises(RuntimeError):
        crawl(failing, paths, limit=100, sleep=no_sleep)
    assert paths.partial.exists() and not paths.final.exists()

    get_page, calls = fake_feed(pages)
    manifest = crawl(get_page, paths, limit=100, sleep=no_sleep)

    assert calls == ["2"]
    assert [j["title"] for j in read_snapshot(paths)] == [f"Job {n}" for n in range(1, 6)]
    assert manifest["pages"] == 3


def test_resume_drops_lines_written_after_last_saved_cursor(paths):
    get_page, _ = fake_feed([[job(1)], [job(2)]])

    def failing(cursor):
        if cursor == "1":
            raise RuntimeError("stop")
        return get_page(cursor)

    with pytest.raises(RuntimeError):
        crawl(failing, paths, limit=100, sleep=no_sleep)
    # Simulate a crash after writing page 2's job but before saving the state.
    with paths.partial.open("a") as f:
        f.write(json.dumps(job(2)) + "\n")

    manifest = crawl(get_page, paths, limit=100, sleep=no_sleep)

    assert [j["title"] for j in read_snapshot(paths)] == ["Job 1", "Job 2"]
    assert manifest["duplicates_skipped"] == 0


def test_http_retries_rate_limit_then_succeeds():
    responses = iter([
        httpx.Response(429, headers={"Retry-After": "7"}),
        httpx.Response(503),
        httpx.Response(200, json={"jobs": [], "nextCursor": None}),
    ])
    seen = []
    client = httpx.Client(transport=httpx.MockTransport(lambda req: seen.append(req) or next(responses)))
    waits = []

    page = http_get_page(client, sleep=waits.append)("abc")

    assert page == {"jobs": [], "nextCursor": None}
    assert waits == [7.0, 4.0]
    assert seen[0].url.params["cursor"] == "abc"
    assert seen[0].url.params["limit"] == "20"


def test_http_does_not_retry_client_errors():
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(404)))

    with pytest.raises(httpx.HTTPStatusError):
        http_get_page(client, sleep=no_sleep)(None)
