#!/usr/bin/env bash

set -e

TOPIC=${1:-panoptes-pipeline-image}
BASE_TAG=$(git rev-parse HEAD)
PROJECT_ID=panoptes-exp

echo "Building image"
gcloud builds submit --tag "gcr.io/${PROJECT_ID}/${TOPIC}:${BASE_TAG}" .

#gcloud compute instances update-container \
#  pipeline-processing \
#  --zone us-central1-a \
#  --container-image "gcr.io/${PROJECT_ID}/${TOPIC}:${BASE_TAG}"

echo "Deploying ${TOPIC} to Cloud Run"
gcloud run deploy "${TOPIC}" \
  --region "us-west1" \
  --image "gcr.io/${PROJECT_ID}/${TOPIC}:${BASE_TAG}" \
  --no-allow-unauthenticated \
  --platform managed \
  --cpu 2 \
  --memory "8Gi" \
  --max-instances 500 \
  --concurrency 1 \
  --timeout "20m"

echo "Deploying ${TOPIC/image/observation} to Cloud Run"
gcloud run deploy "${TOPIC/image/observation}" \
  --region "us-west1" \
  --image "gcr.io/${PROJECT_ID}/${TOPIC}:${BASE_TAG}" \
  --no-allow-unauthenticated \
  --platform managed \
  --cpu 2 \
  --memory "8Gi" \
  --max-instances 50 \
  --concurrency 1 \
  --timeout "20m"
