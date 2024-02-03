FROM debian:11-slim AS pipeline-base
LABEL description="Development environment for working with the PIPELINE"
LABEL maintainers="developers@projectpanoptes.org"
LABEL repo="github.com/panoptes/panoptes-pipeline"

ARG userid=1000
ENV USERID $userid

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED True

ENV PORT 8080

# Install system packages.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget sudo git \
        astrometry.net source-extractor dcraw exiftool \
        libcfitsio-dev libcfitsio-bin \
        libpng-dev libjpeg-dev \
        libfreetype6-dev \
        libffi-dev && \
    # Cleanup
    apt-get autoclean --yes && \
    apt-get --yes clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Add the user to the sudo file and don't ask for password.
    useradd -ms /bin/bash "${USERID}" && \
    echo "${USERID} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-custom-user

ADD http://data.astrometry.net/4100/index-4108.fits /usr/share/astrometry
ADD http://data.astrometry.net/4100/index-4110.fits /usr/share/astrometry
ADD http://data.astrometry.net/4100/index-4111.fits /usr/share/astrometry
ADD http://data.astrometry.net/4100/index-4112.fits /usr/share/astrometry
ADD http://data.astrometry.net/4100/index-4113.fits /usr/share/astrometry
ADD http://data.astrometry.net/4100/index-4114.fits /usr/share/astrometry
ADD http://data.astrometry.net/4100/index-4115.fits /usr/share/astrometry
ADD http://data.astrometry.net/4100/index-4116.fits /usr/share/astrometry
ADD http://data.astrometry.net/4100/index-4117.fits /usr/share/astrometry
ADD http://data.astrometry.net/4100/index-4118.fits /usr/share/astrometry
ADD http://data.astrometry.net/4100/index-4119.fits /usr/share/astrometry

# Install Miniforge and mamba.
RUN wget --no-check-certificate -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash Miniforge3.sh -b -p /opt/conda && \
    chown -R "${USERID}:${USERID}" /opt/conda/ && \
    rm Miniforge3.sh && \
    /opt/conda/bin/conda init bash && \
    /opt/conda/bin/conda clean -afy && \
    /opt/conda/bin/conda update -n base conda && \
    /opt/conda/bin/conda install -c conda-forge mamba

FROM pipeline-base AS panoptes-pipeline

RUN echo "Creating /input and /output directories" && \
    mkdir -p /app && \
    mkdir -p /input && \
    mkdir -p /output && \
    chown -R "${USERID}:${USERID}" /app && \
    chown -R "${USERID}:${USERID}" /input && \
    chown -R "${USERID}:${USERID}" /output && \
    chmod -R 777 /input && \
    chmod -R 777 /output && \
    chmod +r /usr/share/astrometry/*

# Install the Python packages.
USER "${USERID}"
COPY --chown="${USERID}:${USERID}" env.yaml /tmp/env.yaml
RUN /opt/conda/bin/mamba env update -n base -f /tmp/env.yaml && \
    /opt/conda/bin/conda clean -afy && \
    /opt/conda/bin/pip cache purge && \
    sudo apt-get autoclean --yes && \
    sudo apt-get --yes clean && \
    sudo rm -rf /var/lib/apt/lists/*

# Install the PANOPTES Pipeline.
WORKDIR /app
COPY --chown="${USERID}:${USERID}" . /app/
RUN /opt/conda/bin/pip install -e . && \
    rm -rf /app/.git && \
    /opt/conda/bin/pip cache purge

# Set the default command.
ENTRYPOINT ["/opt/conda/bin/gunicorn"]
CMD ["--workers", "1", "--threads", "8", "--timeout", "0", "-k", "uvicorn.workers.UvicornWorker", "--bind", ":8080", "panoptes.pipeline.services.processing:app"]
