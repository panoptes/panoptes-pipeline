#!/usr/bin/env bash

set -e

TOPIC=${1:-panoptes-pipeline-prepare}
BASE_TAG=${1:-latest}
PROJECT_ID=panoptes-exp

echo "Building image"
gcloud builds submit --tag "gcr.io/${PROJECT_ID}/${TOPIC}:${BASE_TAG}" .

echo "Deploying to Cloud Run"
gcloud run deploy "${TOPIC}" \
  --image "gcr.io/${PROJECT_ID}/${TOPIC}:${BASE_TAG}" \
  --no-allow-unauthenticated \
  --platform managed \
  --memory "4Gi" \
  --region "us-west1"
