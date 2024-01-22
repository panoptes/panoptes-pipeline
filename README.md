# panoptes-pipeline

PANOPTES Image Processing for the Extraction of Lightcurves in Nearby Exoplanets

## Description

This repository contains code that allows for the processing and discovering of [PANOPTES](https://www.projectpanoptes.org) data.

Most processing is done in the cloud using dedicated Google Cloud Platform services, but the code contained in this repository can be used to process results locally.

In general the processing is set up as series of Jupyter Notebooks that are processed automatically using [papermill](https://papermill.readthedocs.io/en/latest/). The processed notebooks are stored as artifcats in storage buckets.
