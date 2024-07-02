# panoptes-pipeline

PANOPTES Image Processing for the Extraction of Lightcurves in Nearby Exoplanets

## Description

So many planets, such little time...


## Deployment on  GCP

```shell
gcloud builds submit --config cloudbuild.yaml
```

Then change the sha id in the `pipeline-service.yaml` file, then

```shell
gcloud run services replace pipeline-service.yaml
```
