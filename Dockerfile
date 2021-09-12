ARG image_url=gcr.io/panoptes-exp/panoptes-pocs
ARG image_tag=develop
FROM ${image_url}:${image_tag} AS pipeline-base

LABEL description="Development environment for working with the PIPELINE"
LABEL maintainers="developers@projectpanoptes.org"
LABEL repo="github.com/panoptes/panoptes-pipeline"

ARG userid=1000
ENV USERID $userid

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED True

ENV PORT 8080

USER "${userid}"

USER "${USERID}"
WORKDIR /build
COPY --chown="${USERID}:${USERID}" . .
RUN echo "Building wheel" && \
    sudo chown -R "${userid}:${userid}" /build && \
    python setup.py bdist_wheel -d /build/dist

FROM pipeline-base AS panoptes-pipeline

USER "${USERID}"
WORKDIR /build
COPY --from=pipeline-base /build/dist/ /build/dist
RUN echo "Installing module" && \
    pip install --no-cache-dir "$(ls /build/dist/*.whl)" && \
    # Cleanup
    pip cache purge && \
    conda clean -fay && \
    sudo apt-get autoremove --purge --yes \
        gcc pkg-config git && \
    sudo apt-get autoclean --yes && \
    sudo apt-get --yes clean && \
    sudo rm -rf /var/lib/apt/lists/*

USER "${USERID}"
WORKDIR /app
COPY --chown="${USERID}:${USERID}" ./services/* /app/
COPY ./notebooks/ProcessFITS.ipynb .
COPY ./notebooks/ProcessObservation.ipynb .

RUN echo "Creating /input and /output directories" && \
    sudo mkdir -p /input && \
    sudo mkdir -p /output && \
    sudo chown -R "${USERID}:${USERID}" /input && \
    sudo chown -R "${USERID}:${USERID}" /output && \
    sudo chmod -R 777 /input && \
    sudo chmod -R 777 /output

CMD [ "gunicorn --workers 1 --threads 8 --timeout 0 -k uvicorn.workers.UvicornWorker --bind :${PORT:-8080} pipeline:app" ]
