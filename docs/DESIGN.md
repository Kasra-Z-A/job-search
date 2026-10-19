# Design

How the job search works, how it is evaluated and why it is built this way. Usage
is in the [README](../README.md).

## Goal

Describe the job you want in your own words, for example *"senior data roles open
to people in Germany, at least $90k, full time"*, get matching remote postings with
reasons, and see measured numbers for how well that works compared with keyword and
semantic search.

## Data

**Source: [Himalayas](https://himalayas.app/jobs/api).** Of the job APIs checked
(Himalayas, Arbeitnow, RemoteOK, Remotive, Adzuna), it is the only one with full
descriptions *and* structured fields on nearly every posting: seniority on all,
employment type, country restrictions, categories and a numeric salary on about a
third. Every displayed job links to its original posting and credits Himalayas,
and the data is never passed on to other job boards.

**Frozen snapshot for evaluation.** `job_search.snapshot` pages through the feed
(20 jobs per request, 1 s apart, retries on 429 and 5xx, resumable) and writes a
dated file. The evaluation uses `himalayas-2026-10-13`: 10,000 postings from about
52 hours. The data files stay out of Git; each snapshot's manifest, with job count,
timing and sha256, is committed so a copy can be checked.

**Normalisation** (`job_search.normalise`), which every later step reads:

- **Salary** to yearly USD with the ECB reference rates of the snapshot date
  (`data/fx/`). Hourly × 2080, weekly × 52, monthly × 12. Status `stated`, `none`,
  `unconvertible` (no ECB rate) or `implausible` (below $5k or above $1M a year,
  nearly always monthly pay labelled annual). Only `stated` salaries are labels.
- **Seniority:** Himalayas' levels, deduplicated and ordered.
- **Countries:** an empty list means worldwide.
- **Text:** the HTML description as plain text.
- **English only.** The language is detected from the first 2,000 characters
  (`lingua`); 542 non-English postings are dropped, leaving 9,458. Queries are in
  English, so a matching German posting would measure language, not search.

**Unknown is not "no".** A posting with no stated salary or category is neither a
correct nor a wrong answer for a query on that field and is left out of scoring
for it.

A spot-check of 48 seniority tags (`data/review/`) agreed with the tag 88% of the
time: Entry-level to Senior hold up, Executive is weakest. Seniority is usable as a
label.

## Systems

All systems return a ranking of up to 100 job IDs per query.

| Name | What it does |
|---|---|
| `bm25` | Keyword search over title and description: stemmed words, stopwords removed. |
| `embeddings` | Cosine similarity of OpenAI `text-embedding-3-small` vectors. |
| `hybrid` | Reciprocal rank fusion of the two (K = 60, untuned). |
| `filtered` | The main system. Claude Haiku 5.5 turns the query into filters chosen from the snapshot's own field values (structured output with enums). Jobs that contradict a filter are dropped; known matches rank before jobs where the field is not stated, then by embedding similarity. |
| `textonly` | Same as `filtered`, but on fields Claude Haiku 5.5 extracted from each posting's title and description, never Himalayas' structured fields. |

**Answering.** The live demo runs `filtered`, then Claude Haiku 5.5 reads the top 20
candidates (fields, which requirements they meet, the start of each description)
and returns a short summary and up to six picks with reasons. The model returns job
numbers, not job text: the page renders the cards from the data, picks outside the
candidates are dropped, and a summary or reason that mentions a dollar amount not
in the data is replaced. If the call fails, the ranking is shown instead.

**Follow-ups.** Later messages go to a separate prompt that gets the earlier turns
and returns the full updated filters plus a standalone search sentence for the
embedding. Follow-ups are not part of the evaluation.

## Evaluation

All evaluation runs on the frozen snapshot and frozen, dated query sets; nothing is
regenerated in place. Comparisons between systems use a paired bootstrap over
queries (10,000 resamples, fixed seed). A difference whose 95% interval includes
zero is not a result.

### 1. Labels as the answer key

Queries are generated from the postings' structured fields, so each query's correct
answers are known before any system runs:

```
fields:  {seniority: Senior, category: Data Science, country: Germany, salary_min ≥ 90k USD/yr}
query:   "senior data roles open to people in Germany, at least 90k"
answers: every posting whose fields match
```

1,500 field combinations with 2–4 constraints were sampled from real postings.
Claude Sonnet 5.5 wrote them as queries in three styles (keywords, sentence,
casual), and Claude Haiku 5.5 checked each query against its fields. 1,464 passed
and were frozen; the harness recomputes every answer set and checks it against the
count and hash frozen with the query.

| System | precision@10 | nDCG@10 | recall@100 |
|---|---|---|---|
| random | 0.008 | 0.008 | 0.011 |
| bm25 | 0.113 | 0.120 | 0.113 |
| embeddings | 0.174 | 0.182 | 0.186 |
| hybrid | 0.181 | 0.196 | 0.176 |
| textonly | 0.410 | 0.424 | 0.355 |
| filtered | 0.993 | 0.993 | 0.992 |

**This test is circular for `filtered`.** It filters on the same fields, with the
same rules, that the answer key is built from, so its 0.99 measures query parsing
(98.0% of queries parsed fully correctly) rather than relevance. It also cannot
judge whether a job fits the *topic* of a request within a category. It remains
useful for the systems that never see the labels: text-only extraction more than
doubles precision over embeddings, and most of its remaining gap is information the
descriptions do not contain (only 36–50% state employment type or location).

### 2. Relevance graded by an LLM judge

To measure what the product claims, good matches for realistic requests, a second
query set adds a topic to the constraints ("senior backend Python, fintech, EU,
90k+"): 309 queries, written by Claude Sonnet 5.5 from postings and checked by
Claude Haiku 5.5. The top 10 of each system are pooled (10,967 query–job pairs) and
Claude Haiku 5.5 grades every pair 0 (no), 1 (partial) or 2 (good fit) from the
query, the listing details and the full description, against a written guideline.

| System | nDCG@10 |
|---|---|
| bm25 | 0.242 |
| hybrid | 0.340 |
| embeddings | 0.344 |
| textonly | 0.452 |
| filtered | 0.802 |

`filtered` against `embeddings`: +0.458 [0.423, 0.493]. At least one good match
in the top 10: 94% of queries for `filtered`, 67% for `embeddings`. Hybrid fusion
does not beat embeddings alone (−0.004 [−0.028, 0.019]). `filtered` stays near 0.80
across query styles and constraint counts, while the baselines fall as constraints
are added.

**Checking the judge** (`judge checks`):

- Claude Sonnet 5.5 re-graded 300 pairs: 83% exact agreement, 99.7% within one
  grade, weighted kappa 0.87. Sonnet is stricter, so absolute scores are likely a
  little high.
- Haiku without the listing details: kappa 0.71, with grades drifting to 1, so
  the details decide the requirement checks.
- 100 random query–job pairs: 96 graded 0, 4 graded 1, none 2.

### Limitations

- The queries are LLM-written. 40 deliberately sloppy queries (typos,
  abbreviations) parsed 39 of 40 correctly, but no human-written queries were
  tested.
- There is no hand-graded subset; agreement is measured between models only.
- The judge sees Himalayas' listing details, the same labels `filtered` uses, which
  may favour it on requirement checks.

## Live demo

https://jobs.kasra.page searches a rolling window of the last seven days instead
of the frozen snapshot. The live window is never scored.

- **Refresh** (`job_search.refresh`, twice a day): crawls the feed newest first
  until a whole page is already known, normalises with the latest ECB rates, embeds
  only new postings and drops those older than seven days or expired. It refuses to
  publish a bundle under 1,000 jobs or under half the previous one.
- **Bundles** (`app.store`) are written to S3 under a version prefix, files first
  and the `current.json` pointer last, keeping four versions; rollback is
  rewriting the pointer.
- **Search API:** a Lambda function behind CloudFront. It loads the current bundle
  on a cold start, rereads the pointer at most every five minutes, and reads
  descriptions from S3 by byte range for the 20 candidates only. Answers are
  cached per query; uncached searches are capped per day.
- **One image:** CI tests the code and builds one container image for both
  functions, pushed to ECR; `deploy/deploy.sh` points them at a digest.

The page (`web/`) is plain HTML, CSS and JavaScript with no build step, in the
same design as kasra.page.

## Key decisions

- **Evaluate on a frozen snapshot, serve a live window.** Results must be
  reproducible, and postings expire; the demo uses the same pipeline that was
  evaluated, on current jobs.
- **The LLM judge replaced the label-based test as the headline.** The label test
  cannot be fair to a system that filters on labels; graded relevance can compare
  all systems on the same footing.
- **Structured filters before semantic ranking.** On the label test, embeddings and
  hybrid stay below 0.2 precision@10 because they miss hard constraints like salary,
  country and seniority; parsing them into filters is what moves the score.
- **No hybrid fusion in the main system.** It gave no consistent gain over
  embeddings alone: on the label test it only reordered the top, and on the
  judged queries the difference is zero.
- **Cards from data, not free text.** The model picks and explains; it never writes
  job details, so it cannot invent a posting or a salary.
- **A container image instead of a zip.** The language detector alone is about
  300 MB, too large for a Lambda zip.
