FROM gcr.io/google_appengine/python

RUN virtualenv /env -p python3.4
ENV VIRTUAL_ENV /env
ENV PANUSER panoptes
ENV PANDIR $VIRTUAL_ENV/panoptes
ENV PANLOG $PANDIR/logs 
ENV POCS $PANDIR/POCS
ENV PIAA $PANDIR/PIAA
ENV PAWS $PANDIR/PAWS
ENV PATH /env/bin:$PATH
#ENV PATH $PATH:/usr/bin/python3.4

#RUN git clone https://github.com/panoptes/POCS.git $VIRTUAL_ENV/$POCS 

RUN apt-get update && apt-get install -y libncurses5-dev
ADD requirements.txt /app/requirements.txt
#ADD requirements_pocs.txt /app/requirements_pocs.txt
RUN pip install -r /app/requirements.txt
#RUN pip install -r /app/requirements_pocs.txt
#RUN pip install git+https://github.com/panoptes/POCS.git
RUN git clone https://github.com/panoptes/POCS.git $POCS
RUN pip install -r $POCS/requirements.txt
RUN pip install -e $POCS

ADD . /app

RUN pip install -e /app

#CMD python3.4 /app/scripts/notification_listener.py

EXPOSE 8080
ENTRYPOINT ["python3.4", "/app/scripts/notification_listener.py"]
