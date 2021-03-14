ARG image_url=gcr.io/panoptes-exp/panoptes-utils
ARG image_tag=develop
FROM ${image_url}:${image_tag} AS base

LABEL description="Development environment for working with the PIPELINE"
LABEL maintainers="developers@projectpanoptes.org"
LABEL repo="github.com/panoptes/panoptes-pipeline"

ARG userid=1000
ENV USERID $userid

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

USER "${userid}"
COPY docker/environment.yaml .
RUN /conda/bin/conda env update -n base -f environment.yaml

ARG pip_install_name="."
ARG pip_install_extras=""
ARG port=8080
ENV PORT=port

COPY --chown="${userid}:${userid}" . .
RUN echo "Installing ${pip_install_name} module with ${pip_install_extras}" && \
    /conda/bin/pip install "${pip_install_name}${pip_install_extras}" && \
#     /conda/bin/jupyter labextension install @pyviz/jupyterlab_pyviz && \
    # Cleanup
    /conda/bin/pip cache purge && \
    /conda/bin/conda clean -fay && \
    sudo apt-get autoremove --purge --yes \
        gcc pkg-config git && \
    sudo apt-get autoclean --yes && \
    sudo apt-get --yes clean && \
    sudo rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENTRYPOINT [ "/usr/bin/env", "bash", "-ic" ]
CMD [ "/conda/bin/jupyter lab --port ${PORT}" ]
