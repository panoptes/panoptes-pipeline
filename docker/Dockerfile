ARG arch=amd64

FROM gcr.io/panoptes-survey/panoptes-utils:$arch

ARG user=root
ARG pan_dir=/var/panoptes
ARG env_name=panoptes-env

ENV PANDIR $pan_dir
ENV PIAA ${PANDIR}/PIAA

ENV USER $user
ENV CONDA_ENV $env_name

USER ${USER}

COPY . ${PIAA}
WORKDIR ${PIAA}

RUN cd ${PIAA} && \
    /${USER}/conda/envs/${CONDA_ENV}/bin/pip install -r requirements.txt && \
    /${USER}/conda/envs/${CONDA_ENV}/bin/pip install -e .

# This Dockerfile will mostly be used as another layer.
CMD ["/bin/zsh"]

