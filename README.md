# job-search

Search remote job postings by describing the job in your own words. An LLM turns
the request into filters (seniority, salary, country, employment type, category),
the search keeps only postings that meet them and ranks the rest by meaning, and
the answer names the best matches with reasons. Follow-up messages refine the
search.

Live demo: https://jobs.kasra.page

## Results

Measured on a frozen snapshot of 9,458 English postings from
[Himalayas](https://himalayas.app). An LLM judge graded the top 10 results of each
system for 309 realistic requests with a topic, such as "senior backend Python,
fintech, EU, 90k+":

| System | nDCG@10 |
|---|---|
| Keyword search (BM25) | 0.242 |
| Keyword + semantic (hybrid) | 0.340 |
| Semantic search (embeddings) | 0.344 |
| Labels extracted from the text | 0.452 |
| **This system: parsed filters + semantic ranking** | **0.802** |

For 94% of requests its top 10 contains a good match, against 67% for semantic
search. The judge agrees with a stronger model on 83% of grades (99.7% within one grade).
How the evaluation works, what it cannot show and why the system is built this
way: [docs/DESIGN.md](docs/DESIGN.md).

## Run it

Python 3.12 and [uv](https://docs.astral.sh/uv/). LLM and embedding steps need
`ANTHROPIC_API_KEY` and `OPENAI_API_KEY` and are billable.

```sh
uv sync
uv run pytest -q

# Data
uv run python -m job_search.snapshot --limit 10000   # dated snapshot in data/snapshots/
uv run python -m job_search.normalise data/snapshots/himalayas-2026-10-13.jsonl.gz

# Demo, locally
uv run python -m job_search.app.bundle data/normalised/himalayas-2026-10-13.jsonl.gz   # demo data + web/results.json
uv run python -m job_search.app.devserver data/app/2026-10-13

# Live 7-day window, into a folder or the S3 bucket (--bucket)
uv run python -m job_search.refresh --dir data/live
```

### Evaluation

```sh
uv run python -m job_search.specs data/normalised/himalayas-2026-10-13.jsonl.gz
uv run python -m job_search.queries sample data/eval/specs-2026-10-13.jsonl -n 20
uv run python -m job_search.systems.bm25 data/normalised/himalayas-2026-10-13.jsonl.gz
uv run python -m job_search.systems.embeddings data/normalised/himalayas-2026-10-13.jsonl.gz
uv run python -m job_search.systems.hybrid data/normalised/himalayas-2026-10-13.jsonl.gz
uv run python -m job_search.systems.parse data/normalised/himalayas-2026-10-13.jsonl.gz
uv run python -m job_search.systems.filtered data/normalised/himalayas-2026-10-13.jsonl.gz
uv run python -m job_search.systems.filtered data/normalised/himalayas-2026-10-13.jsonl.gz --fields extracted   # text-only
uv run python -m job_search.evaluate score data/runs/<system>-2026-10-13.jsonl
uv run python -m job_search.evaluate compare data/runs/<a>-2026-10-13.jsonl data/runs/<b>-2026-10-13.jsonl
uv run python -m job_search.judge run data/normalised/himalayas-2026-10-13.jsonl.gz
uv run python -m job_search.judge score data/normalised/himalayas-2026-10-13.jsonl.gz
```

Each module's docstring lists its subcommands. Frozen query sets, parses, grades
and scores are committed under `data/`; snapshots, runs and embeddings are not.

### Deploy

CI tests every push and, on `main`, builds one container image for the search API
and the refresh job and pushes it to ECR. The infrastructure is defined in a
separate Terraform repository.

```sh
deploy/deploy.sh sha256:...               # digest from the CI run summary
deploy/deploy.sh --web-only               # page changes only
```

Job data from [Himalayas](https://himalayas.app); every result links to the
original posting.
