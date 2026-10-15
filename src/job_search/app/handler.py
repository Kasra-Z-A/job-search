"""AWS Lambda entry point for the demo's search API.

CloudFront forwards `/api/*` to the function URL; the function is not
reachable without CloudFront.

- `GET /api/search?q=...` answers the first message of a conversation. The
  answer depends only on q, so it is cacheable for a day: repeated queries
  (the page's examples) reach the function, and cost money, once.
- `GET /api/followup?q=...&ctx=...` answers a later message. ctx is the
  conversation so far, base64url-encoded JSON: up to four turns of
  {q, filters, picks (job ids)}. These answers are never cached.

Each uncached message costs about $0.002 in API calls (Haiku twice, one
embedding). A DynamoDB counter
caps searches per UTC day (DAILY_LIMIT); after that the API answers 429
until midnight. API keys are read from SSM once per container.

Environment: BUNDLE_DIR, ANTHROPIC_KEY_PARAM, OPENAI_KEY_PARAM, DAILY_TABLE,
DAILY_LIMIT.
"""

from __future__ import annotations

import base64
import binascii
import datetime as dt
import json
import os
import time
from pathlib import Path

DAY_SECONDS = 86_400
_searcher = None


def response(status: int, body: dict, cache: bool = False) -> dict:
    return {
        "statusCode": status,
        "headers": {
            "content-type": "application/json",
            "cache-control": f"public, max-age={DAY_SECONDS}" if cache else "no-store",
        },
        "body": json.dumps(body, ensure_ascii=False),
    }


class DailyCounter:
    """Atomic per-day count in DynamoDB; refuses once the limit is reached."""

    def __init__(self, table, limit: int):
        self.table, self.limit = table, limit

    def take(self, today: str | None = None) -> bool:
        from botocore.exceptions import ClientError

        today = today or dt.datetime.now(dt.UTC).date().isoformat()
        try:
            self.table.update_item(
                Key={"day": today},
                UpdateExpression="ADD searches :one SET expires_at = :expires",
                ConditionExpression="attribute_not_exists(searches) OR searches < :limit",
                ExpressionAttributeValues={":one": 1, ":limit": self.limit,
                                           ":expires": int(time.time()) + 7 * DAY_SECONDS},
            )
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise


MAX_CTX_CHARS = 6000


def decode_history(ctx: str) -> list[dict] | None:
    """The conversation sent back by the page, or None if it is malformed."""
    if not ctx or len(ctx) > MAX_CTX_CHARS:
        return None
    try:
        turns = json.loads(base64.urlsafe_b64decode(ctx + "=" * (-len(ctx) % 4)))
    except (binascii.Error, ValueError):
        return None
    if not isinstance(turns, list) or not 1 <= len(turns) <= 4:
        return None
    out = []
    for t in turns:
        if not (isinstance(t, dict) and isinstance(t.get("q"), str) and isinstance(t.get("filters"), dict)
                and isinstance(t.get("picks"), list) and all(isinstance(p, str) for p in t["picks"])):
            return None
        out.append({"q": t["q"], "filters": t["filters"], "picks": t["picks"][:10]})
    return out


def handle(params: dict, searcher, counter, followup: bool = False) -> dict:
    """The API without Lambda specifics, so it can be tested and served locally."""
    from job_search.app.search import MAX_QUERY_CHARS

    query = " ".join((params.get("q") or "").split())
    if not 3 <= len(query) <= MAX_QUERY_CHARS:
        return response(400, {"error": "bad_query", "message": f"Use 3 to {MAX_QUERY_CHARS} characters."})
    history = None
    if followup:
        history = decode_history(params.get("ctx") or "")
        if history is None:
            return response(400, {"error": "bad_context", "message": "Start a new search."})
    if counter is not None and not counter.take():
        return response(429, {"error": "daily_limit",
                              "message": "Today's search budget is used up. Try again tomorrow."})
    try:
        return response(200, searcher.search(query, history), cache=not followup)
    except Exception as exc:  # an upstream API failing should not look like a page error
        print(f"search failed: {type(exc).__name__}: {exc}")
        return response(502, {"error": "upstream", "message": "Search is unavailable right now."})


def _setup():
    import anthropic
    import boto3
    import openai

    from job_search.app.search import Searcher

    ssm = boto3.client("ssm")

    def secret(name):
        return ssm.get_parameter(Name=os.environ[name], WithDecryption=True)["Parameter"]["Value"]

    searcher = Searcher(
        Path(os.environ.get("BUNDLE_DIR", "bundle")),
        # Bounded so a slow upstream fails within CloudFront's 30 s origin timeout.
        anthropic.Anthropic(api_key=secret("ANTHROPIC_KEY_PARAM"), max_retries=1, timeout=8),
        openai.OpenAI(api_key=secret("OPENAI_KEY_PARAM"), max_retries=1, timeout=5),
    )
    table = boto3.resource("dynamodb").Table(os.environ["DAILY_TABLE"])
    return searcher, DailyCounter(table, int(os.environ.get("DAILY_LIMIT", "300")))


def lambda_handler(event, context):
    global _searcher
    path = event.get("rawPath", "")
    if path.endswith("/api/health"):
        return response(200, {"ok": True})
    if not path.endswith(("/api/search", "/api/followup")):
        return response(404, {"error": "not_found"})
    if _searcher is None:
        _searcher = _setup()
    searcher, counter = _searcher
    return handle(event.get("queryStringParameters") or {}, searcher, counter,
                  followup=path.endswith("/api/followup"))
