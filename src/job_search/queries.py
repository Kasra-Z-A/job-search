"""Turn query specs into natural-language queries and check them.

Two LLM stages, each run through the Message Batches API (half price):

1. generate: Claude Sonnet 5.5 writes search queries the way job seekers type
   them, GROUP_SIZE specs per request so it can vary wording across the list
   (one query per request came out as "I am looking for..." every time). Each
   spec gets a style (keywords, sentence or casual) in rotation, so styles are
   balanced and recorded.
2. check: Claude Haiku 5.5 sees one spec and its query and decides whether
   the query asks for exactly those constraints. It verifies against the spec
   instead of parsing the query blind, so the kept queries are not biased
   towards ones an LLM finds easy to parse.

`freeze` keeps the specs whose query passed and writes the evaluation set,
after two fixed rules for ambiguities the checker does not judge (found by
reading the 2026-10-13 run):

- a salary query naming a country other than the United States must mark the
  amount as dollars ("$", "USD", "dollar"); "jobs in Canada 100k+" reads as
  Canadian dollars, while the label is US dollars;
- a query for a country that is also a US state (Georgia) is dropped: on a
  board where most jobs are US-only, "I'm based in Georgia" makes US jobs
  correct answers too.
Raw model outputs are kept in `data/eval/raw/` for provenance.

    uv run python -m job_search.queries sample data/eval/specs-2026-10-13.jsonl -n 20
    uv run python -m job_search.queries generate data/eval/specs-2026-10-13.jsonl
    uv run python -m job_search.queries check data/eval/specs-2026-10-13.jsonl
    uv run python -m job_search.queries freeze data/eval/specs-2026-10-13.jsonl

`generate` and `check` submit a batch, remember its id, and wait for it;
rerunning resumes waiting on the same batch instead of submitting again.
The API key comes from ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

GENERATOR_MODEL = "claude-sonnet-5-5"
CHECKER_MODEL = "claude-haiku-5-5"
MAX_TOKENS = 8000  # thinking counts towards it; the answers themselves are short
GROUP_SIZE = 10
STYLES = ["keywords", "sentence", "casual"]
RAW_DIR = Path("data/eval/raw")
COUNTRIES_NAMED_LIKE_US_STATES = {"Georgia"}
DOLLAR_MARK = re.compile(r"\$|\busd\b|dollar", re.I)

# Shared by writer and checker, so both accept the same wording for a value.
VOCABULARY = """\
Vocabulary (each value and the words that mean the same):
- Seniority. Entry-level: junior, entry level, graduate, new grad. Mid-level:
  mid-level, intermediate. Senior: senior, sr. Manager: manager, people
  manager, management role. Leadership: director, head of, VP, executive.
- Job category, a broad job family; any common name for that family counts.
  Developer: software, developer, engineering (software) jobs. Data Science:
  data, analytics, ML jobs. Content Creator: content, writing, copywriting
  jobs. Customer Service: customer service, support jobs. Human Resources:
  HR, people, recruiting jobs. Other categories by their usual names.
- Employment type. Full Time: full-time, FT. Part Time: part-time.
  Contractor: contract, freelance. Intern: internship, intern.
- Open to candidates in X: the job can hire someone living in X ("I'm based
  in X", "open to people in X", "hiring in X", "X-based").
- Minimum yearly salary: the exact amount in US dollars ("at least $120k",
  "120k+", "$120,000 or more"). Rounded, different or approximate amounts
  ("around 120k", "100-150k") do not count."""

GENERATOR_SYSTEM = f"""\
You write search queries for a remote job board, the way real job seekers type
them. You get a numbered list of items, each with constraints and a style, and
you write one query per item that asks for exactly that item's constraints.

Express every constraint, with the same value. Add no other requirement: no
skills, tools, industries, benefits or company types. Every job on the board
is remote, so "remote" may appear but adds nothing.

{VOCABULARY}

Styles:
- keywords: a short search-box string, no sentence ("sr data jobs canada 140k+").
- sentence: one complete request or question a job seeker writes ("Show me...",
  "Which...", "Find..."). Never a statement that describes a job, like an ad.
- casual: how someone types to a friend or a chatbot, first person, relaxed.

Real people rarely write "I am looking for". Across the list, vary openings,
word order and sentence structure; no two queries should start the same way."""

CHECKER_SYSTEM = f"""\
You check search queries for a remote job board against the constraints they
were written from. Decide whether the query asks for exactly these
constraints: every constraint expressed with the same value, and no other
requirement.

{VOCABULARY}

Mismatch:
- a constraint that is missing, or whose value changed: a different level,
  category, country or employment type, or a salary that is not the exact
  amount;
- an added requirement: a skill, tool, industry, benefit, company type,
  years of experience beyond what the seniority implies, another country.

Not a mismatch: the word "remote" (every job is remote), any wording from the
vocabulary for the same value, or casual phrasing.

First write the problem (or "none"), then the verdict that follows from it."""

QUERIES_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"item": {"type": "integer"}, "query": {"type": "string"}},
                "required": ["item", "query"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["queries"],
    "additionalProperties": False,
}
CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "problem": {"type": "string"},
        "verdict": {"type": "string", "enum": ["match", "mismatch"]},
    },
    "required": ["problem", "verdict"],
    "additionalProperties": False,
}

FIELD_LABELS = {
    "seniority": "seniority",
    "category": "job category",
    "country": "open to candidates in",
    "employment_type": "employment type",
    "salary_min_usd": "minimum yearly salary (USD)",
}


# --- prompts --------------------------------------------------------------


def describe(constraints: dict, indent: str = "") -> str:
    lines = []
    for field, value in constraints.items():
        shown = f"${value:,}" if field == "salary_min_usd" else value
        lines.append(f"{indent}- {FIELD_LABELS[field]}: {shown}")
    return "\n".join(lines)


def style_for(spec_id: str) -> str:
    return STYLES[int(spec_id.lstrip("s")) % len(STYLES)]


def groups(specs: list[dict], size: int = GROUP_SIZE) -> dict[str, list[dict]]:
    return {f"g{i // size + 1:04d}": specs[i:i + size] for i in range(0, len(specs), size)}


def generate_params(group: list[dict]) -> dict:
    items = [
        f"{n}. style: {style_for(spec['spec_id'])}\n{describe(spec['constraints'], '   ')}"
        for n, spec in enumerate(group, 1)
    ]
    return {
        "model": GENERATOR_MODEL,
        "max_tokens": MAX_TOKENS,
        "system": GENERATOR_SYSTEM,
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": QUERIES_SCHEMA}},
        "messages": [{"role": "user", "content": "\n\n".join(items)}],
    }


def split_group(group: list[dict], output: dict) -> dict[str, str]:
    """Spec id -> query, for the items the model answered."""
    by_item = {q["item"]: q["query"].strip() for q in output["queries"]}
    return {spec["spec_id"]: by_item[n] for n, spec in enumerate(group, 1) if by_item.get(n)}


def check_params(spec: dict, query: str) -> dict:
    prompt = f"Constraints:\n{describe(spec['constraints'])}\n\nQuery: {query}"
    return {
        "model": CHECKER_MODEL,
        "max_tokens": MAX_TOKENS,
        "system": CHECKER_SYSTEM,
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": CHECK_SCHEMA}},
        "messages": [{"role": "user", "content": prompt}],
    }


def parse_output(message) -> dict:
    """The JSON object from a structured-output response."""
    if message.stop_reason != "end_turn":
        raise ValueError(f"stop_reason {message.stop_reason}")
    text = next(block.text for block in message.content if block.type == "text")
    return json.loads(text)


# --- files ----------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def stage_paths(specs_path: Path, stage: str, raw_dir: Path = RAW_DIR) -> tuple[Path, Path]:
    stem = specs_path.stem.replace("specs-", "")
    return raw_dir / f"{stage}-{stem}.jsonl", raw_dir / f".{stage}-{stem}.batch.json"


# --- batch runs -----------------------------------------------------------


def run_batch(client: anthropic.Anthropic, requests: dict[str, dict], state: Path,
              poll: float = 60, sleep=time.sleep) -> tuple[list[dict], dict]:
    """Submit (or resume) one batch; return one row per request and token usage."""
    if state.exists():
        batch_id = json.loads(state.read_text())["batch_id"]
        print(f"resuming batch {batch_id}", file=sys.stderr)
    else:
        batch = client.messages.batches.create(requests=[
            Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**params))
            for cid, params in requests.items()
        ])
        batch_id = batch.id
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({"batch_id": batch_id}) + "\n")
        print(f"submitted batch {batch_id} with {len(requests)} requests", file=sys.stderr)

    while (batch := client.messages.batches.retrieve(batch_id)).processing_status != "ended":
        counts = batch.request_counts
        print(f"  {batch.processing_status}: {counts.processing} processing, {counts.succeeded} done",
              file=sys.stderr)
        sleep(poll)

    rows, usage = [], collections.Counter()
    for item in client.messages.batches.results(batch_id):
        row = {"custom_id": item.custom_id, "status": item.result.type}
        if item.result.type == "succeeded":
            message = item.result.message
            usage["input_tokens"] += message.usage.input_tokens
            usage["output_tokens"] += message.usage.output_tokens
            try:
                row["output"] = parse_output(message)
            except (ValueError, StopIteration) as exc:
                row["status"] = f"unparsed: {exc}"
        rows.append(row)
    rows.sort(key=lambda r: r["custom_id"])
    return rows, dict(usage)


def _finish(rows: list[dict], usage: dict, results: Path, state: Path) -> None:
    write_jsonl(results, rows)
    state.unlink(missing_ok=True)
    print(json.dumps({"results": str(results), "rows": len(rows),
                      "status": dict(collections.Counter(r["status"] for r in rows)), **usage}, indent=2))


def generate(client, specs_path: Path, raw_dir: Path = RAW_DIR, **kw) -> None:
    results, state = stage_paths(specs_path, "generate", raw_dir)
    if results.exists():
        raise FileExistsError(f"{results} exists; raw outputs are never overwritten")
    grouped = groups(load_jsonl(specs_path))
    group_rows, usage = run_batch(client, {gid: generate_params(g) for gid, g in grouped.items()}, state, **kw)

    rows = []
    for g in group_rows:
        answered = split_group(grouped[g["custom_id"]], g["output"]) if g["status"] == "succeeded" else {}
        for spec in grouped[g["custom_id"]]:
            sid = spec["spec_id"]
            if sid in answered:
                rows.append({"custom_id": sid, "group": g["custom_id"], "status": "succeeded",
                             "output": {"query": answered[sid]}})
            else:
                rows.append({"custom_id": sid, "group": g["custom_id"],
                             "status": g["status"] if g["status"] != "succeeded" else "missing"})
    _finish(sorted(rows, key=lambda r: r["custom_id"]), usage, results, state)


def check(client, specs_path: Path, raw_dir: Path = RAW_DIR, **kw) -> None:
    results, state = stage_paths(specs_path, "check", raw_dir)
    if results.exists():
        raise FileExistsError(f"{results} exists; raw outputs are never overwritten")
    specs = {s["spec_id"]: s for s in load_jsonl(specs_path)}
    generated, _ = stage_paths(specs_path, "generate", raw_dir)
    queries = {r["custom_id"]: r["output"]["query"] for r in load_jsonl(generated) if r["status"] == "succeeded"}
    rows, usage = run_batch(client, {sid: check_params(specs[sid], q) for sid, q in queries.items()}, state, **kw)
    _finish(rows, usage, results, state)


# --- freeze ---------------------------------------------------------------


def ambiguity(constraints: dict, query: str) -> str | None:
    country = constraints.get("country")
    if country in COUNTRIES_NAMED_LIKE_US_STATES:
        return "country is also a US state"
    if "salary_min_usd" in constraints and country not in (None, "United States") and not DOLLAR_MARK.search(query):
        return "currency unclear"
    return None


def freeze(specs_path: Path, raw_dir: Path = RAW_DIR) -> dict:
    specs = load_jsonl(specs_path)
    generated = {r["custom_id"]: r for r in load_jsonl(stage_paths(specs_path, "generate", raw_dir)[0])}
    checked = {r["custom_id"]: r for r in load_jsonl(stage_paths(specs_path, "check", raw_dir)[0])}

    kept, dropped = [], collections.Counter()
    for spec in specs:
        sid = spec["spec_id"]
        gen, chk = generated.get(sid), checked.get(sid)
        if not gen or gen["status"] != "succeeded":
            dropped["generation failed"] += 1
        elif not chk or chk["status"] != "succeeded":
            dropped["check failed"] += 1
        elif chk["output"]["verdict"] != "match":
            dropped["mismatch"] += 1
        elif reason := ambiguity(spec["constraints"], gen["output"]["query"]):
            dropped[reason] += 1
        else:
            kept.append({**spec, "query": gen["output"]["query"], "style": style_for(sid)})

    out = specs_path.with_name(specs_path.name.replace("specs-", "queries-"))
    if out.exists():
        raise FileExistsError(f"{out} exists; the evaluation set is frozen")
    write_jsonl(out, kept)
    return {
        "output": str(out),
        "kept": len(kept),
        "dropped": dict(dropped),
        "styles": dict(collections.Counter(q["style"] for q in kept)),
        "constraints_per_query": dict(sorted(collections.Counter(len(q["constraints"]) for q in kept).items())),
    }


# --- sample (direct calls, for reading before a full run) -----------------


def sample(client, specs_path: Path, n: int) -> list[dict]:
    """Whole groups from evenly spaced points in the spec file, so all styles appear."""
    specs = load_jsonl(specs_path)
    all_groups = list(groups(specs).values())
    wanted = max(1, n // GROUP_SIZE)
    picked = all_groups[:: max(1, len(all_groups) // wanted)][:wanted]
    rows = []
    for group in picked:
        answered = split_group(group, parse_output(client.messages.create(**generate_params(group))))
        for spec in group:
            query = answered.get(spec["spec_id"])
            if query is None:
                print(f"{spec['spec_id']} missing from the model's answer", file=sys.stderr)
                continue
            verdict = parse_output(client.messages.create(**check_params(spec, query)))
            rows.append({"spec_id": spec["spec_id"], "style": style_for(spec["spec_id"]),
                         "constraints": spec["constraints"], "query": query, **verdict})
            print(f"{spec['spec_id']} [{rows[-1]['style']}] {verdict['verdict']}: {query}", file=sys.stderr)
            if verdict["verdict"] != "match":
                print(f"    problem: {verdict['problem']}", file=sys.stderr)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["sample", "generate", "check", "freeze"])
    parser.add_argument("specs", type=Path)
    parser.add_argument("-n", type=int, default=20, help="specs for `sample`")
    args = parser.parse_args(argv)

    if args.command == "freeze":
        print(json.dumps(freeze(args.specs), indent=2))
        return
    client = anthropic.Anthropic()
    if args.command == "sample":
        rows = sample(client, args.specs, args.n)
        out = RAW_DIR / f"sample-{args.specs.stem.replace('specs-', '')}.jsonl"
        write_jsonl(out, rows)
        print(out)
    elif args.command == "generate":
        generate(client, args.specs)
    else:
        check(client, args.specs)


if __name__ == "__main__":
    main()
