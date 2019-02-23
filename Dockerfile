FROM gcr.io/panoptes-survey/pocs-base:latest

ENV PIAA $PANDIR/PIAA  

COPY . $PIAA

RUN wget https://github.com/panoptes/panoptes-network/archive/master.tar.gz && \
	mv master.tar.gz $PANDIR && \
	cd $PANDIR && \
	tar zxvf master.tar.gz && \
	mv panoptes-network-master panoptes-network && \
	cd $PIAA && \
	pip3 install -Ur requirements.txt && \
	pip3 install -e .

WORKDIR ${PIAA}

# This Dockerfile will mostly be used as another layer.
CMD ["/bin/bash"]
