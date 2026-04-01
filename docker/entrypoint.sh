#!/bin/bash
set -e

if [ -n "$ARTIFACTS_S3_BUCKET" ] && [ "$ENV" != "local" ]; then
    echo ">>> Fetching artifacts from S3: $ARTIFACTS_S3_BUCKET ..."
    mkdir -p /app/artifacts/feature_engineering /app/artifacts/model_training

    ENDPOINT_FLAG=""
    if [ -n "$AWS_ENDPOINT_URL" ]; then
        ENDPOINT_FLAG="--endpoint-url $AWS_ENDPOINT_URL"
    fi

    aws s3 sync s3://${ARTIFACTS_S3_BUCKET}/artifacts/feature_engineering/ /app/artifacts/feature_engineering/ --exact-timestamps $ENDPOINT_FLAG || echo "Warning: Feature eng sync failed"
    aws s3 sync s3://${ARTIFACTS_S3_BUCKET}/artifacts/model_training/ /app/artifacts/model_training/ --exact-timestamps $ENDPOINT_FLAG || echo "Warning: Model sync failed"
    echo ">>> Artifacts ready."
else
    echo ">>> Skipping S3 artifact fetch (using local bind mounts or no bucket specified)."
fi

echo ">>> Starting application..."
exec "$@"
