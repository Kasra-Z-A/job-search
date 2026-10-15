#!/usr/bin/env bash
# Deploy jobs.kasra.page: build the Lambda package, upload it, point the
# function at it, then sync the page and clear CloudFront's cache.
# Infrastructure comes from the aws-infra repo's jobs-site stack.
#
#   AWS_PROFILE=admin deploy/deploy.sh 2026-10-13
#   AWS_PROFILE=admin deploy/deploy.sh 2026-10-13 --web-only   # page only, no Lambda build
set -euo pipefail

date="${1:?snapshot date, e.g. 2026-10-13}"
web_only="${2:-}"
root="$(cd "$(dirname "$0")/.." && pwd)"
region="eu-central-1"
account="$(aws sts get-caller-identity --query Account --output text)"
site_bucket="jobs-kasra-page-$account"
artifacts_bucket="jobs-kasra-page-artifacts-$account"
function="job-search"
distribution="$(aws cloudfront list-distributions \
  --query "DistributionList.Items[?contains(Aliases.Items, 'jobs.kasra.page')].Id" --output text)"

if [ "$web_only" != "--web-only" ]; then
  "$root/deploy/build_lambda.sh" "$date"
  key="lambda/job-search-$(date -u +%Y%m%dT%H%M%SZ).zip"
  aws s3 cp --only-show-errors --region "$region" "$root/build/lambda.zip" "s3://$artifacts_bucket/$key"
  aws lambda update-function-code --region "$region" --function-name "$function" \
    --s3-bucket "$artifacts_bucket" --s3-key "$key" --query LastUpdateStatus --output text
  aws lambda wait function-updated-v2 --region "$region" --function-name "$function"
  echo "deployed $key"
fi

# Long cache for nothing: the page is small and changes with each deploy.
aws s3 sync --only-show-errors --region "$region" --delete "$root/web/" "s3://$site_bucket/" \
  --cache-control "public, max-age=300"
aws cloudfront create-invalidation --distribution-id "$distribution" --paths "/*" \
  --query Invalidation.Id --output text
echo "page synced"
