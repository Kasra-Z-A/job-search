"""Serve the demo page and its API locally, without a daily limit.

API keys come from ANTHROPIC_API_KEY and OPENAI_API_KEY.

    uv run python -m job_search.app.devserver data/app/2026-10-13
"""

from __future__ import annotations

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import anthropic
import openai

from job_search.app.handler import handle
from job_search.app.search import Searcher

WEB_DIR = Path("web")


class Handler(SimpleHTTPRequestHandler):
    searcher: Searcher

    def do_GET(self):
        url = urlparse(self.path)
        if url.path != "/api/search":
            return super().do_GET()
        params = {k: v[0] for k, v in parse_qs(url.query).items()}
        result = handle(params, self.searcher, counter=None)
        self.send_response(result["statusCode"])
        for name, value in result["headers"].items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(result["body"].encode())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--port", type=int, default=8800)
    args = parser.parse_args(argv)

    Handler.searcher = Searcher(args.bundle, anthropic.Anthropic(), openai.OpenAI())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), functools.partial(Handler, directory=str(WEB_DIR)))
    print(f"http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
