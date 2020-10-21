FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-2:latest

LABEL description="Development environment for working with the pipeline on GCP notebooks"
LABEL maintainers="developers@projectpanoptes.org"
LABEL repo="github.com/panoptes/panoptes-pipeline:gcp-notebook"

COPY . /var/panoptes/panoptes-pipeline
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        astrometry.net sextractor dcraw exiftool \
        libcfitsio-dev libcfitsio-bin \
        libpng-dev libjpeg-dev \
        libfreetype6-dev \
        libffi-dev && \
    cd /var/panoptes/panoptes-pipeline && \
    pip install -e . && \
    # Cleanup
    apt-get autoremove --purge --yes && \
    apt-get autoclean --yes && \
    rm -rf /var/lib/apt/lists/*
