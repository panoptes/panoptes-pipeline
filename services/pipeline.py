import os
import tempfile
from pathlib import Path
from typing import Tuple, Optional

from fastapi import FastAPI
from google.cloud import firestore
from google.cloud import pubsub
from google.cloud import storage
from panoptes.utils.serializers import from_json
from pydantic import BaseModel, HttpUrl

from panoptes.pipeline.scripts.image import Settings as ImageSettings
from panoptes.pipeline.scripts.image import process_notebook as process_image_notebook
from panoptes.pipeline.scripts.observation import process_notebook as process_observation_notebook
from panoptes.pipeline.utils.gcp.storage import move_blob_to_bucket
from panoptes.pipeline.utils.metadata import ImageStatus, \
    ObservationStatus

app = FastAPI()
storage_client = storage.Client()
firestore_db = firestore.Client()
publisher = pubsub.PublisherClient()

PROJECT_ID = os.getenv('PROJECT_ID', 'panoptes-exp')
EXTRACT_TOPIC = os.getenv('EXTRACT_STAMP_TOPIC', 'extract-stamps')
ROOT_URL = os.getenv('PUBLIC_URL_BASE', 'https://storage.googleapis.com')
INPUT_NOTEBOOK = os.getenv('INPUT_NOTEBOOK', 'ProcessFITS.ipynb')
processed_bucket = storage_client.get_bucket(
    os.getenv('OUTPUT_BUCKET', 'panoptes-images-processed'))
incoming_bucket = storage_client.get_bucket(os.getenv('INPUT_BUCKET', 'panoptes-images-incoming'))
error_bucket = storage_client.get_bucket(os.getenv('ERROR_BUCKET', 'panoptes-images-error'))
extract_stamp_topic = f'projects/{PROJECT_ID}/topics/{EXTRACT_TOPIC}'


class ObservationInfo(BaseModel):
    sequence_id: str
    frame_slice: Tuple[Optional[int], Optional[int]] = (None, None)
    stamp_size: Tuple[int, int] = (10, 10)
    base_url: HttpUrl = 'https://storage.googleapis.com/panoptes-images-processed'
    image_filename: Path = 'image.fits.fz'
    source_filename: Path = 'sources.parquet'
    image_status: ImageStatus = ImageStatus.MATCHED
    force_new: bool = False


@app.post('/image/process')
def process_image_from_pubsub(message_envelope: dict):
    print(f'Received {message_envelope}')

    message = message_envelope['message']
    bucket = message['attributes']['bucketId']
    bucket_path = message['attributes']['objectId']
    public_bucket_path = f'{ROOT_URL}/{bucket}/{bucket_path}'
    image_settings = ImageSettings(output_dir='temp',
                                   **from_json(message['attributes'].get('imageSettings', '{}')))

    process_image(public_bucket_path, image_settings)


@app.post('/image/process/notebook')
def process_image(bucket_path, image_settings: ImageSettings):
    with tempfile.TemporaryDirectory() as tmp_dir:
        image_settings.output_dir = tmp_dir
        try:
            process_image_notebook(bucket_path, Path(tmp_dir), upload=True)
            return_dict = {'success': True}
        except FileExistsError as e:
            print(f'Skipping already processed file.')
            return_dict = {'success': False, 'error': f'{e!r}'}
        except Exception as e:
            print(f'Problem processing image for {bucket_path}: {e!r}')
            return_dict = {'success': False, 'error': f'{e!r}'}

            # Move to error bucket.
            try:
                new_blob = move_blob_to_bucket(bucket_path, processed_bucket, error_bucket)
                return_dict['error_bucket_path'] = new_blob.path
            except Exception as e2:
                print(
                    f'Error moving {bucket_path} to {error_bucket} from {incoming_bucket}: {e2!r}')
                return_dict['error_2'] = f'{e2!r}'

        # Success.
        return return_dict


class Observation(BaseModel):
    sequence_id: str


@app.post('/observation/process')
def process_observation_from_pubsub(message_envelope: dict):
    print(f'Received {message_envelope}')

    message = message_envelope['message']
    sequence_id = message['attributes']['sequenceId']

    process_observation(Observation(sequence_id=sequence_id))


@app.post('/observation/process/notebook')
def process_observation(observation: Observation):
    sequence_id = observation.sequence_id
    print(f'Received {sequence_id=}')
    unit_id, camera_id, sequence_time = sequence_id.split('_')

    # Get sequence information
    sequence_doc_ref = firestore_db.document(f'units/{unit_id}/observations/{sequence_id}')
    if sequence_doc_ref.get().exists is False:
        return dict(success=False, message=f'No record for {sequence_id}')

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            process_observation_notebook(sequence_id, output_dir=Path(tmp_dir),
                                         process_images=True, upload=True)
            return_dict = {'success': True}
        except FileExistsError as e:
            print(f'Skipping already processed observation {sequence_id}')
            return_dict = {'success': False, 'error': f'{e!r}'}
        except Exception as e:
            print(f'Problem processing image for {sequence_id}: {e!r}')
            return_dict = {'success': False, 'error': f'{e!r}'}

        # Success.
        return return_dict
