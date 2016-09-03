PIAA
====

[![Build Status](https://travis-ci.org/panoptes/PIAA.svg?branch=master)](https://travis-ci.org/panoptes/PIAA)

The Panoptes Image Analysis Algorithm (PIAA) repository contains the data processing pipeline and algorithms for the images taken by PANOPTES observatories. PIAA currently handles storing the raw data, processing intermediate data and storing final data products. Eventually it will also analyze final data products and store results. 

For a detailed explanation of the system, design choices and alternatives, see the [Design Doc](https://docs.google.com/document/d/1GefBo-vYa6jKhT8LO7Z-fd7rppOMjD7AIczNJLBhS-o/edit#heading=h.xgjl2srtytjt).

## System Components

####Data Simulator####
Produces simulated light curves and Postage Stamp Cubes (PSCs) and uploads them to a Google Cloud Storage (GCS) bucket.
####App Engine Notification Proxy####
Receives notifications when new objects are added to the GCS bucket and pushes them to the Kubernetes cluster.
####Kubernetes Cluster####
Cluster of nodes managed using Kubernetes on Google Container Engine (GKE) and Docker. A Flask server on each pod recieves notifications from App Engine and spawns subprocesses to combine the stored light curves into master light curves for each Panoptes Input Catalog (PIC) star.
####Google Cloud Storage####
All simulated data inputs and products - light curves, PSCs and master light curves - are currently stored in a GCS bucket.


## Running the Light Curve Analysis

### Setting Up the Notification Channel

GCS has a feature called Object Change Notifications. These send an HTTP request with the metadata of the changed object in a given bucket. To set up this channel, runr
~~~
gsutil notification watchbucket -i <channel-name> <app-url> gs://<bucket-name>
~~~
Currently, a channel is set up called `panoptes-simulated-data-channel` that sends notifications to  `https://notification-proxy-dot-panoptes-survey.appspot.com/`, an App Engine proxy, when changes occur in the bucket `panoptes-simulated-data`.

### Deploying the App Engine Proxy

Clone the source code from the [Google Cloud repository](https://pantheon.corp.google.com/code/develop/browse/notification-proxy/master?project=panoptes-survey) into a local directory named `notification-proxy`. Make sure the [App Engine SDK](https://cloud.google.com/appengine/downloads#Google_App_Engine_SDK_for_Python) is installed. Deploy using 
~~~
appcfg.py update -A panoptes-survey .
~~~

### Kubernetes/GKE Cluster

#### Updating the Docker image

In your local PIAA directory, make desired changes. Make sure [kubectl](http://kubernetes.io/docs/user-guide/prereqs/), [Docker](https://docs.docker.com/) and the [Cloud SDK](https://cloud.google.com/sdk/docs/quickstarts) are installed. Then run
~~~
docker build -t gcr.io/panoptes-survey/piaa:<version> .
gcloud docker push gcr.io/panoptes-survey/piaa:<version>
kubectl set image deployment/combiner piaa=gcr.io/panoptes-survey/piaa:<version>
~~~
where `<version>` is the new version number of the image. The existing versions can be found on [Google Container Registry](https://pantheon.corp.google.com/kubernetes/images/tags/piaa?location=GLOBAL&project=panoptes-survey).

#### Resizing the Kubernetes Cluster

To change the number of nodes in the cluster, run
~~~
gcloud container clusters resize <cluster_name> --size <num_nodes>
~~~
where `<cluster_name>` is the GKE cluster, currently `image-analysis-cluster`.

To change the number of replicated pods that run on the nodes, run
~~~
kubectl scale deployment/<deployment_name> --replicas=<num_pods>
~~~
where `<deployment_name>` is the name of the Deployment, currently `combiner`.


### Development 

#### Setting up

Clone this repository to a directory named PIAA on your local machine and run
~~~
pip install -r requirements.txt
~~~
to install the dependencies. Set up everything as in the [Coding in PANOPTES wiki](https://github.com/panoptes/POCS/wiki/Coding-in-PANOPTES), and add the environment variable  
~~~
PIAA=$PANDIR/PIAA
~~~

#### Running on Google Compute Engine

For development purposes, a Google Compute Engine (GCE) instance can be used. The instance piaa-instance has been used for development and contains the PIAA and POCS repos, though they likely need to be updated. The url that App Engine sends notifications to needs to be changed to the external IP for the instance (set `listener='GCE'`). 

#### Logging

The logs can be monitored to see how system components are running. Here is where to find the [logs for the App Engine notification proxy](https://pantheon.corp.google.com/logs/viewer?project=panoptes-survey&minLogLevel=0&expandAll=false&resource=appengine.googleapis.com%2Fmodule_id%2Fnotification-proxy&key1=notification-proxy&logName=&advancedFilter=metadata.serviceName%3D%22appengine.googleapis.com%22%0Ametadata.labels.%22appengine.googleapis.com%2Fmodule_id%22%3D%22notification-proxy%22%0A) and the [logs for the Container Engine cluster](https://pantheon.corp.google.com/logs/viewer?project=panoptes-survey&minLogLevel=0&expandAll=false&resource=container.googleapis.com&key1=notification-proxy&logName=&lastVisibleTimestampNanos=1472781565000000000).
