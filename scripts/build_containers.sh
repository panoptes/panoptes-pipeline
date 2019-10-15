#!/bin/bash -e
SOURCE_DIR=${PANDIR}/PIAA

echo "Building panoptes-piaa Docker GCE instance"
gcloud builds submit \
    --timeout="5h" \
    --config "${SOURCE_DIR}/docker/cloudbuild.yaml" \
    ${SOURCE_DIR}

