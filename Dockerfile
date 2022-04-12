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
    astrometry.net libcfitsio-bin git

ADD http://data.astrometry.net/4100/index-4108.fits /usr/share/astrometry
ADD http://data.astrometry.net/4100/index-4110.fits /usr/share/astrometry
ADD http://data.astrometry.net/4100/index-4111.fits /usr/share/astrometry
ADD http://data.astrometry.net/4100/index-4112.fits /usr/share/astrometry
ADD http://data.astrometry.net/4100/index-4113.fits /usr/share/astrometry
ADD http://data.astrometry.net/4100/index-4114.fits /usr/share/astrometry
ADD http://data.astrometry.net/4100/index-4115.fits /usr/share/astrometry
ADD http://data.astrometry.net/4100/index-4116.fits /usr/share/astrometry
ADD http://data.astrometry.net/4100/index-4117.fits /usr/share/astrometry
ADD http://data.astrometry.net/4100/index-4118.fits /usr/share/astrometry
ADD http://data.astrometry.net/4100/index-4119.fits /usr/share/astrometry

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
