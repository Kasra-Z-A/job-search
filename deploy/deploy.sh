#!/usr/bin/env bash
# Deploy jobs.kasra.page: point both Lambda functions (search API and refresh)
# at an image CI built and pushed, then sync the page and clear CloudFront's
# cache. Infrastructure comes from the aws-infra repo's jobs-site stack; the
# jobs themselves come from the refresh job, not from deploys.
#
#   AWS_PROFILE=admin deploy/deploy.sh <account>.dkr.ecr.eu-central-1.amazonaws.com/job-search@sha256:...
#   AWS_PROFILE=admin deploy/deploy.sh --web-only   # page only
#
# CI prints the digest in the run summary. Rolling back is deploying an older one.
set -euo pipefail

image="${1:?image digest from CI, or --web-only}"
root="$(cd "$(dirname "$0")/.." && pwd)"
region="eu-central-1"
account="$(aws sts get-caller-identity --query Account --output text)"
site_bucket="jobs-kasra-page-$account"
distribution="$(aws cloudfront list-distributions \
  --query "DistributionList.Items[?contains(Aliases.Items, 'jobs.kasra.page')].Id" --output text)"

if [ "$image" != "--web-only" ]; then
  case "$image" in
    *@sha256:*) ;;
    *) echo "pin a digest (…@sha256:…), not a tag: $image" >&2; exit 1 ;;
  esac
  for function in job-search job-search-refresh; do
    aws lambda update-function-code --region "$region" --function-name "$function" \
      --image-uri "$image" --query LastUpdateStatus --output text
    aws lambda wait function-updated-v2 --region "$region" --function-name "$function"
  done
  echo "deployed $image"
fi

# Long cache for nothing: the page is small and changes with each deploy.
aws s3 sync --only-show-errors --region "$region" --delete "$root/web/" "s3://$site_bucket/" \
  --cache-control "public, max-age=300"
aws cloudfront create-invalidation --distribution-id "$distribution" --paths "/*" \
  --query Invalidation.Id --output text
echo "page synced"
