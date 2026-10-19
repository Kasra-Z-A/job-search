# job-search

Search remote job postings by describing the job in your own words, measured
against an answer key built from the postings' own fields.

Live demo: https://jobs.kasra.page. Plan, results and decisions: [docs/PLAN.md](docs/PLAN.md).

```sh
uv sync
uv run pytest -q
uv run python -m job_search.snapshot --limit 10000   # dated snapshot in data/snapshots/
uv run python -m job_search.normalise data/snapshots/himalayas-2026-10-13.jsonl.gz
uv run python -m job_search.app.bundle data/normalised/himalayas-2026-10-13.jsonl.gz   # local demo data + web/results.json
uv run python -m job_search.refresh --dir data/live   # live 7-day window (OPENAI_API_KEY; S3 with --bucket)
AWS_PROFILE=admin deploy/deploy.sh <ecr-image@sha256:...>   # digest CI printed for a push to main
AWS_PROFILE=admin deploy/deploy.sh --web-only   # page changes only
uv run python -m job_search.spotcheck render data/review/seniority-spotcheck-2026-10-13.csv
```

The evaluation runs on a frozen snapshot (2026-10-13). The live site searches
a rolling window of the last seven days, which a scheduled Lambda refreshes
twice a day; CI builds one image for it and the search API.

Job data from [Himalayas](https://himalayas.app); every result links to the
original posting.
