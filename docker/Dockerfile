ARG arch=amd64

FROM gcr.io/panoptes-survey/panoptes-utils:latest

ENV PIAA ${PANDIR}/PIAA

COPY . ${PIAA}
WORKDIR ${PIAA}

RUN cd ${PIAA} && \
    pip install -r requirements.txt && \
    pip install -e .

# This Dockerfile will mostly be used as another layer.
CMD ["/bin/zsh"]

