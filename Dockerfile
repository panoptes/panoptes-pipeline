FROM gcr.io/panoptes-survey/panoptes-utils:latest

ENV PIAA ${PANDIR}/PIAA

COPY . ${PIAA}

RUN cd ${PIAA} && \
	pip install -r --no-deps requirements.txt && \
	pip install -e .

WORKDIR ${PIAA}

# This Dockerfile will mostly be used as another layer.
CMD ["/bin/bash"]
