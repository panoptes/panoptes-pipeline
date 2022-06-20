FROM debian:11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        astrometry.net source-extractor dcraw exiftool \
        libcfitsio-dev libcfitsio-bin \
        libpng-dev libjpeg-dev \
        libfreetype6-dev \
        libffi-dev && \
    # Cleanup
    apt-get autoremove --purge --yes && \
    apt-get autoclean --yes && \
    rm -rf /var/lib/apt/lists/*

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

RUN conda update -n base conda && \
    conda init && \
    conda create -n pipeline -c conda-forge \
        astropy astroplan astroquery photutils \
        scipy numpy pandas scikit-learn scikit-image numexpr \
        bokeh seaborn plotly panel \
        jupyterlab ipywidgets ipython-autotime \
        gcsfs google-cloud-storage \
        h5py \
        pip
