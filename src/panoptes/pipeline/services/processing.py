import os
import tempfile
from pathlib import Path

from fastapi import FastAPI
from google.cloud import firestore, storage
from panoptes.utils.serializers import from_json
from pydantic import ValidationError

from panoptes.pipeline.image import process_notebook as process_image_notebook
from panoptes.pipeline.observation import process_notebook as process_observation_notebook
from panoptes.pipeline.settings import ImageSettings, ObservationSettings

app = FastAPI()
storage_client = storage.Client()
firestore_db = firestore.Client()

FITS_NOTEBOOK = os.getenv('INPUT_NOTEBOOK', '/app/notebooks/ProcessFITS.ipynb')
OBS_NOTEBOOK = os.getenv('INPUT_NOTEBOOK', '/app/notebooks/ProcessObservation.ipynb')

incoming_bucket = storage_client.get_bucket(os.getenv('INPUT_BUCKET', 'panoptes-images-incoming'))
processed_bucket = storage_client.get_bucket(os.getenv('OUTPUT_BUCKET', 'panoptes-processed-images'))
error_bucket = storage_client.get_bucket(os.getenv('ERROR_BUCKET', 'panoptes-images-error'))


@app.post('/image/process')
def process_image_from_pubsub(envelope: dict):
    print(f'Received {envelope}')
    message = envelope['message']
    attributes = message['attributes']

    response = dict(success=False)
    try:
        public_url = attributes['public_url']
    except KeyError:
        print(f'Missing bucket_path in message attributes.')
        return response

    image_settings = ImageSettings(**from_json(attributes.get('imageSettings', '{}')))

    try:
        response = process_image(public_url, image_settings)
        print(f'Finished processing {public_url} with {response}')
        response['success'] = True
    except Exception as e:
        print(f'Problem with processing from pubsub notification: {e}')

    return response


@app.post('/image/process/notebook')
def process_image(bucket_path, image_settings: ImageSettings):
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            print(f'Processing {bucket_path=} with {image_settings}')

            public_url_list = process_image_notebook(
                bucket_path,
                input_notebook=Path(FITS_NOTEBOOK),
                image_settings=image_settings,
                output_dir=Path(tmp_dir),
            )

            return_dict = {'success': True, 'urls': public_url_list}
        except FileExistsError as e:
            print(f'Skipping already processed file.')
            return_dict = {'success': False, 'error': f'{e!r}'}
        except Exception as e:
            print(f'Problem processing image for {bucket_path}: {e!r}')
            return_dict = {'success': False, 'error': f'{e!r}'}
        finally:
            print(f'Finished processing for {bucket_path} in {tmp_dir!r}')

    # Return the status and any other relevant info.
    return return_dict


@app.post('/observation/process')
def process_observation_from_pubsub(envelope: dict):
    print(f'Received {envelope}')
    message = envelope['message']
    attributes = message['attributes']

    print(f'Received {attributes=}')
    response = dict(success=False)

    # Build the observation processing params from the attributes. Must include a sequence_id.
    try:
        params = ObservationSettings(**attributes)
        response = process_observation(params)
    except ValidationError:
        print(f'Missing sequence_id param.')
    finally:
        return response


@app.post('/observation/process/notebook')
def process_observation(params: ObservationSettings):
    sequence_id = params.sequence_id
    print(f'Received {params=}')

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            public_url_list = process_observation_notebook(
                sequence_id,
                input_notebook=Path(OBS_NOTEBOOK),
                fits_notebook=Path(FITS_NOTEBOOK),
                output_dir=Path(tmp_dir),
                process_images=params.process_images,
                upload=params.upload,
                force_new=params.force_new
            )
            return_dict = {'success': True, 'urls': public_url_list}
        except FileExistsError as e:
            print(f'Skipping already processed observation {sequence_id}')
            return_dict = {'success': False, 'error': f'{e!r}'}
        except RuntimeError as e:
            print(f'Skipping processing observation {sequence_id}')
            return_dict = {'success': False, 'error': f'{e!r}'}
        except Exception as e:
            print(f'Problem processing observation for {sequence_id}: {e!r}')
            return_dict = {'success': False, 'error': f'{e!r}'}

        # Success.
        return return_dict
