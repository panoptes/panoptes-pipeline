import os
import tempfile
from pathlib import Path
from typing import Tuple, Optional

from fastapi import FastAPI
from google.cloud import firestore
from google.cloud import storage
from panoptes.data.images import ImagePathInfo
from panoptes.utils.serializers import from_json
from pydantic import BaseModel, HttpUrl, ValidationError

from panoptes.pipeline.image import Settings as ImageSettings
from panoptes.pipeline.image import process_notebook
from panoptes.pipeline.scripts.observation import process_notebook as process_observation_notebook
from panoptes.pipeline.utils.gcp.firestore import get_firestore_refs
from panoptes.data.images import ImageStatus

from panoptes.pipeline.utils.gcp.storage import move_blob_to_bucket, upload_dir

app = FastAPI()
storage_client = storage.Client()
firestore_db = firestore.Client()

PROJECT_ID = os.getenv('PROJECT_ID', 'panoptes-project-01')
ROOT_URL = os.getenv('PUBLIC_URL_BASE', 'https://storage.googleapis.com')
INPUT_NOTEBOOK = os.getenv('INPUT_NOTEBOOK', '/app/notebooks/ProcessFITS.ipynb')

incoming_bucket = storage_client.get_bucket(os.getenv('INPUT_BUCKET', 'panoptes-image-processing'))
processed_bucket = storage_client.get_bucket(os.getenv('OUTPUT_BUCKET', 'panoptes-processed-images'))
error_bucket = storage_client.get_bucket(os.getenv('ERROR_BUCKET', 'panoptes-images-error'))


class ObservationInfo(BaseModel):
    sequence_id: str
    frame_slice: Tuple[Optional[int], Optional[int]] = (None, None)
    stamp_size: Tuple[int, int] = (10, 10)
    base_url: HttpUrl = 'https://storage.googleapis.com/panoptes-images-processed'
    image_filename: Path = 'image.fits.fz'
    source_filename: Path = 'sources.parquet'
    image_status: ImageStatus = ImageStatus.MATCHED
    force_new: bool = False


class ObservationParams(BaseModel):
    sequence_id: str
    process_images: bool = True
    upload: bool = True
    force_new: bool = False


@app.post('/image/process')
def process_image_from_pubsub(message: dict):
    print(f'Received {message}')

    response = dict(success=False)
    bucket_path = message['id']
    image_settings = ImageSettings(output_dir='temp', **from_json(message.get('imageSettings', '{}')))

    try:
        response = process_image(bucket_path, image_settings)
        response['success'] = True
    except Exception as e:
        print(f'Problem with processing from pubsub notification: {e}')

    return response


@app.post('/image/process/notebook')
def process_image(bucket_path, image_settings: ImageSettings, upload: bool = True):
    try:
        print(f'Checking if got a fits file at {bucket_path}')
        path_info = ImagePathInfo(path=bucket_path)
        print(f'Got image info: {path_info}')
    except ValueError as e:
        raise RuntimeError(f'Need a FITS file, got {bucket_path}')

    unit_doc_ref, seq_doc_ref, image_doc_ref = get_firestore_refs(bucket_path)

    try:
        image_status = image_doc_ref.get(['status']).to_dict()['status']
        print(f'Current status for {bucket_path} is {ImageStatus[image_status].name}')
    except (KeyError, ValueError):
        print(f'No status found for {bucket_path}, setting to {ImageStatus.UNKNOWN.name}')
        image_status = ImageStatus.UNKNOWN.name

    if ImageStatus[image_status] >= ImageStatus.PROCESSING:
        print(f'Skipping image with status of {image_status}')
        return dict(success=False, error=f'Skipping image with status of {image_status}')

    # Update the image status.
    print(f'Updating status for {bucket_path} from {image_status} to {ImageStatus.PROCESSING.name}')
    image_doc_ref.set({'status': ImageStatus.PROCESSING.name}, merge=True)

    # Assume we will upload to the processed bucket.
    outgoing_bucket = processed_bucket

    with tempfile.TemporaryDirectory() as output_dir:
        image_settings.output_dir = output_dir
        print(f'Processing {bucket_path=} with {image_settings}')

        try:
            notebook_path, has_errors = process_notebook(path_info,
                                                         Path(INPUT_NOTEBOOK),
                                                         settings=image_settings,
                                                         output_dir=Path(output_dir),
                                                         )

            # If there is an error processing the notebook, it is still generated but with errors.
            if has_errors:
                raise Exception(f'Notebook {notebook_path} had errors.')

            return_dict = {'success': True, 'url_list': notebook_path}
        except FileExistsError as e:
            print(f'Skipping already processed file.')
            return_dict = {'success': False, 'error': f'{e!r}'}
        except Exception as e:
            print(f'Problem processing image for {bucket_path}: {e!r}')
            outgoing_bucket = error_bucket
            image_doc_ref.set({'status': ImageStatus.ERROR.name}, merge=True)
            return_dict = {'success': False, 'error': f'{e!r}'}
        else:
            # If successful, write metadata to firestore and then remove the file.
            try:
                metadata_file = Path(output_dir) / 'metadata.json'
                if metadata_file.exists():
                    with metadata_file.open() as f:
                        image_metadata = from_json(f.read())

                    image_metadata['image']['processed_time'] = firestore.SERVER_TIMESTAMP

                    unit_doc_ref.set(image_metadata['unit'], merge=True)
                    seq_doc_ref.set(image_metadata['sequence'], merge=True)
                    image_doc_ref.set(image_metadata['image'], merge=True)
                    print(f'Recorded metadata for {bucket_path} with {image_doc_ref.id=}')

                    # Remove the metadata file.
                    # metadata_file.unlink()
            except FileNotFoundError:
                raise FileNotFoundError(f'No metadata file found in {image_settings.output_dir}!')
        finally:
            # Move image from the incoming bucket to the processed bucket using the mounted volumes.
            new_blob = move_blob_to_bucket(bucket_path, incoming_bucket, outgoing_bucket)
            if new_blob is not None and hasattr(new_blob, 'name'):
                return_dict['processed_bucket_path'] = new_blob.name

            # Copy any assets to the upload bucket.
            if upload:
                output_url_list = upload_dir(Path(output_dir),
                                             prefix=path_info.get_full_id(sep='/'),
                                             bucket=outgoing_bucket)
                return_dict['output_url_list'] = output_url_list

    print(f'Finished processing for {bucket_path} in {image_settings.output_dir!r}')

    # Return the status and any other relevant info.
    return return_dict


@app.post('/observation/process')
def process_observation_from_pubsub(message: dict):
    print(f'Received {message}')
    response = dict(success=False)

    # Build the observation processing params from the attributes. Must include a sequence_id.
    try:
        params = ObservationParams(**message)
        response = process_observation(params)
    except ValidationError:
        print(f'Missing sequence_id param.')
    finally:
        return response


@app.post('/observation/process/notebook')
def process_observation(params: ObservationParams):
    sequence_id = params.sequence_id
    print(f'Received {params=}')

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            public_url_list = process_observation_notebook(sequence_id,
                                                           output_dir=Path(tmp_dir),
                                                           process_images=params.process_images,
                                                           upload=params.upload,
                                                           force_new=params.force_new
                                                           )
            return_dict = {'success': True, 'urls': public_url_list}
        except FileExistsError as e:
            print(f'Skipping already processed observation {sequence_id}')
            return_dict = {'success': False, 'error': f'{e!r}'}
        except Exception as e:
            print(f'Problem processing image for {sequence_id}: {e!r}')
            return_dict = {'success': False, 'error': f'{e!r}'}

        # Success.
        return return_dict
