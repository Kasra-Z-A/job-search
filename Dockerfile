# One image for both Lambda functions: the search API (default command) and
# the scheduled refresh (`job_search.refresh.lambda_handler`, set in Terraform).
# CI builds it once per push to main; deploys pin its digest.
FROM public.ecr.aws/lambda/python:3.12
RUN pip install --no-cache-dir uv==0.12.10
COPY pyproject.toml uv.lock README.md /build/
# Runtime dependencies plus the pipeline group, which carries the language
# detector the refresh needs (~300 MB, the reason this is an image and not a
# zip). boto3 comes with the Lambda runtime.
RUN uv export --project /build --locked --no-default-groups --group pipeline --no-hashes \
      --no-emit-project --format requirements-txt -q > /build/requirements.txt \
 && uv pip install --no-cache -r /build/requirements.txt --target "${LAMBDA_TASK_ROOT}" \
 && rm -rf /build
COPY src/job_search ${LAMBDA_TASK_ROOT}/job_search
CMD ["job_search.app.handler.lambda_handler"]
