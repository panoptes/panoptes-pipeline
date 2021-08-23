# panoptes-pipeline

PANOPTES Image Processing for the Extraction of Lightcurves in Nearby Exoplanets

## Description

So many planets, such little time...

### Deploy

See [Deployment](../README.md#deploy) in main README for preferred deployment method.

#### Notification Creation

The bucket notification only needs to be created once, which can be done with the following command:

```sh
gsutil notification create -t process-raw-image -f json -e OBJECT_FINALIZE gs://panoptes-images-raw/
```

You can list existing notifications with:

```sh
gsutil notification list gs://panoptes-images-incoming/
```