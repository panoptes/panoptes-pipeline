FROM gcr.io/panoptes-survey/pocs-base:latest

ENV PIAA ${PANDIR}/PIAA  

COPY . ${PIAA}

RUN cd ${PIAA} && \
	pip3 install -Ur requirements.txt && \
	pip3 install -e .

WORKDIR ${PIAA}

# This Dockerfile will mostly be used as another layer.
CMD ["/bin/bash"]
