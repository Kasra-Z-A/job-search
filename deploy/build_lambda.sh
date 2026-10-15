#!/usr/bin/env bash
# Build the search API's Lambda package (Python 3.12, arm64, Amazon Linux 2023,
# so manylinux_2_28 wheels) without Docker:
# uv installs Linux wheels for the runtime dependencies straight from macOS.
#
#   deploy/build_lambda.sh 2026-10-13      -> build/lambda.zip
set -euo pipefail

date="${1:?snapshot date, e.g. 2026-10-13}"
root="$(cd "$(dirname "$0")/.." && pwd)"
build="$root/build"
pkg="$build/lambda"

rm -rf "$pkg" "$build/lambda.zip"
mkdir -p "$pkg"

# Runtime dependencies only: no dev or pipeline groups. boto3 is part of the
# Lambda runtime, so it is not packaged.
uv export --project "$root" --no-default-groups --no-hashes --no-emit-project \
  --format requirements-txt -q > "$build/requirements.txt"
uv pip install -q -r "$build/requirements.txt" --target "$pkg" \
  --python-platform aarch64-manylinux_2_28 --python-version 3.12 --only-binary :all:

cp -R "$root/src/job_search" "$pkg/"
cp -R "$root/data/app/$date" "$pkg/bundle"
find "$pkg" -name "__pycache__" -type d -prune -exec rm -rf {} +

(cd "$pkg" && zip -qr -9 "$build/lambda.zip" .)
du -sh "$pkg" "$build/lambda.zip"
