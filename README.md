# job-search

Search remote job postings by describing the job in your own words, measured
against an answer key built from the postings' own fields.

Early stage: see [docs/PLAN.md](docs/PLAN.md).

```sh
uv sync
uv run pytest -q
uv run python -m job_search.snapshot --limit 10000   # dated snapshot in data/snapshots/
uv run python -m job_search.normalise data/snapshots/himalayas-2026-10-13.jsonl.gz
```

Job data from [Himalayas](https://himalayas.app); every result links to the
original posting.
