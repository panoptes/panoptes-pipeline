ARG image_url=python
ARG image_tag=slim-buster
FROM ${image_url}:${image_tag} AS pipeline-base

LABEL description="Development environment for working with the PIPELINE"
LABEL maintainers="developers@projectpanoptes.org"
LABEL repo="github.com/panoptes/panoptes-pipeline"

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED True

ENV PORT 8080

RUN apt-get update && apt-get install --yes --no-install-recommends \
    astrometry.net astrometry-data-tycho2 git

WORKDIR /build
COPY . .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir "." && mkdir /input /output

WORKDIR /app
COPY ./services/* /app/
COPY ./notebooks/ProcessFITS.ipynb .
COPY ./notebooks/ProcessObservation.ipynb .

ENTRYPOINT [ "/usr/bin/env", "bash", "-ic" ]
CMD [ "gunicorn --workers 1 --threads 8 --timeout 0 -k uvicorn.workers.UvicornWorker --bind :${PORT:-8080} pipeline:app" ]
