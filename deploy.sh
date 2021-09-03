#!/usr/bin/env bash

set -e

TOPIC=${1:-panoptes-pipeline-image}
BASE_TAG=$(git rev-parse HEAD)
PROJECT_ID=panoptes-exp

echo "Building image"
gcloud builds submit --tag "gcr.io/${PROJECT_ID}/${TOPIC}:${BASE_TAG}" .

echo "Deploying to Cloud Run"
gcloud run deploy "${TOPIC}" \
  --region "us-west1" \
  --image "gcr.io/${PROJECT_ID}/${TOPIC}:${BASE_TAG}" \
  --no-allow-unauthenticated \
  --platform managed \
  --cpu 2 \
  --memory "4Gi" \
  --max-instances 25 \
  --concurrency 2 \
  --timeout "10m"
