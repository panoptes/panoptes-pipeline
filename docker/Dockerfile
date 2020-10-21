ARG image_url=gcr.io/panoptes-exp/panoptes-utils:develop
FROM ${image_url} AS pocs-base

LABEL description="Development environment for working with the pipeline"
LABEL maintainers="developers@projectpanoptes.org"
LABEL repo="github.com/panoptes/panoptes-pipeline"

ARG panuser=panoptes
ARG userid=1000
ARG pan_dir=/var/panoptes
ARG pocs_dir="${pan_dir}/POCS"
ARG conda_env_name="base"

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV SHELL /bin/zsh

ENV USERID $userid
ENV PANDIR $pan_dir
ENV PANLOG "$pan_dir/logs"
ENV PANUSER $panuser
ENV POCS $pocs_dir

COPY --chown=panoptes:panoptes . /var/panoptes/panoptes-pipeline/
RUN echo "Installing panoptes-pipeline" && \
    cd /var/panoptes/panoptes-pipeline && \
    "${PANDIR}/conda/bin/pip" install -e . && \
    # Cleanup
    sudo apt-get autoremove --purge --yes && \
    sudo apt-get autoclean --yes && \
    sudo rm -rf /var/lib/apt/lists/*

USER root
WORKDIR ${PANDIR}

