FROM gcr.io/google_appengine/python

# Setup virtualenv and environment variables
RUN virtualenv /env -p python3.4
ENV VIRTUAL_ENV /env
ENV PANUSER panoptes
ENV PANDIR $VIRTUAL_ENV/panoptes
ENV PANLOG $PANDIR/logs 
ENV POCS $PANDIR/POCS
ENV PIAA $PANDIR/PIAA
ENV PAWS $PANDIR/PAWS
ENV PATH /env/bin:$PATH

# Install PIAA dependencies
ADD requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Clone POCS and install dependencies
RUN apt-get update && apt-get install -y \  # extra dependencies to fix install errors
    libncurses5-dev \
    python3-tk
RUN git clone -b develop https://github.com/panoptes/POCS.git $POCS
RUN pip install -r $POCS/requirements.txt
RUN pip install -e $POCS

# Copy PIAA files and install as package
ADD . /app
WORKDIR /app
RUN pip install -e .

# Set entrypoint as the Flask notification listener and expose port
EXPOSE 8080
ENTRYPOINT ["python3.4", "./scripts/notification_listener.py"]
