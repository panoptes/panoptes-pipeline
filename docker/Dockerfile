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

USER "${userid}"
COPY docker/environment.yaml .
RUN conda env update -n base -f environment.yaml

FROM pipeline-base AS pipeline

ARG pip_install_name="."
ARG pip_install_extras=""
ARG port=6565
ENV PORT=$port

WORKDIR /panoptes-pipeline
COPY --chown="${userid}:${userid}" . .
RUN echo "Installing ${pip_install_name} module with ${pip_install_extras}" && \
    sudo chown -R "${userid}:${userid}" /panoptes-pipeline && \
    pip install -e "${pip_install_name}${pip_install_extras}" && \
    # Cleanup
    pip cache purge && \
    conda clean -fay && \
    sudo apt-get autoremove --purge --yes \
        gcc pkg-config git && \
    sudo apt-get autoclean --yes && \
    sudo apt-get --yes clean && \
    sudo rm -rf /var/lib/apt/lists/*

CMD [ "jupyter lab --port ${PORT} --no-browser" ]
