FROM gcr.io/panoptes-survey/pocs:latest

ENV PIAA $PANDIR/PIAA  

RUN rm /bin/sh && ln -s /bin/bash /bin/sh \
    && mkdir -p ${PIAA} \
    && wget --quiet https://github.com/panoptes/PIAA/archive/develop.tar.gz -O PIAA.tar.gz \
    && tar zxf PIAA.tar.gz -C $PIAA --strip-components=1 \
    && rm PIAA.tar.gz \
    && cd $PIAA && /opt/conda/bin/pip install -Ur requirements.txt \
    && /opt/conda/bin/pip install -U setuptools \
    && /opt/conda/bin/python setup.py install \
    && cd $PANDIR \
    && /opt/conda/bin/conda clean --all --yes

WORKDIR ${PIAA}

CMD ["/bin/bash"]