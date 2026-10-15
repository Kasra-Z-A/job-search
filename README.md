# job-search

Search remote job postings by describing the job in your own words, measured
against an answer key built from the postings' own fields.

Live demo: https://jobs.kasra.page. Plan, results and decisions: [docs/PLAN.md](docs/PLAN.md).

```sh
uv sync
uv run pytest -q
uv run python -m job_search.snapshot --limit 10000   # dated snapshot in data/snapshots/
uv run python -m job_search.normalise data/snapshots/himalayas-2026-10-13.jsonl.gz
uv run python -m job_search.app.bundle data/normalised/himalayas-2026-10-13.jsonl.gz   # demo data + web/results.json
AWS_PROFILE=admin deploy/deploy.sh 2026-10-13   # build, upload and deploy jobs.kasra.page
uv run python -m job_search.spotcheck render data/review/seniority-spotcheck-2026-10-13.csv
```

Job data from [Himalayas](https://himalayas.app); every result links to the
original posting.
