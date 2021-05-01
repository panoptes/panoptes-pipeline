ARG image_url=gcr.io/panoptes-exp/panoptes-pocs
ARG image_tag=develop
FROM ${image_url}:${image_tag} AS pipeline

LABEL description="Development environment for working with the PIPELINE"
LABEL maintainers="developers@projectpanoptes.org"
LABEL repo="github.com/panoptes/panoptes-pipeline"

ARG userid=1000
ENV USERID $userid

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

USER "${userid}"

ARG pip_install_name="."
ARG pip_install_extras=""

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

ENTRYPOINT [ "/usr/bin/env", "python", "/panoptes-pipeline/scripts/process-image.py" ]
CMD [ "--help" ]
