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

WORKDIR /build
COPY --chown="${userid}:${userid}" . .
RUN echo "Building wheel" && \
    sudo chown -R "${userid}:${userid}" /build && \
    python setup.py bdist_wheel

FROM ${image_url}:${image_tag} AS panoptes-pipeline

WORKDIR /app
COPY --from=pipeline-base /build/dist/ /build/dist
RUN echo "Installing module" && \
    sudo chown -R "${userid}:${userid}" /app && \
    pip install "$(ls /build/dist/*.whl)" && \
    # Cleanup
    pip cache purge && \
    conda clean -fay && \
    sudo apt-get autoremove --purge --yes \
        gcc pkg-config git && \
    sudo apt-get autoclean --yes && \
    sudo apt-get --yes clean && \
    sudo rm -rf /var/lib/apt/lists/*

CMD [ "gunicorn --workers 1 --threads 8 --timeout 0 -k uvicorn.workers.UvicornWorker --bind :${PORT:-8080} image:app" ]