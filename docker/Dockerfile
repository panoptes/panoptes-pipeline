FROM gcr.io/panoptes-survey/panoptes-utils:latest

ENV PIAA ${PANDIR}/PIAA

COPY . ${PIAA}
WORKDIR ${PIAA}

RUN cd ${PIAA} && pip install .

# This Dockerfile will mostly be used as another layer.
CMD ["/bin/zsh"]

