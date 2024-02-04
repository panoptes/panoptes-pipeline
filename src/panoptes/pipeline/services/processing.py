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

app = FastAPI()
storage_client = storage.Client()
firestore_db = firestore.Client()

PROJECT_ID = os.getenv('PROJECT_ID', 'panoptes-project-01')
ROOT_URL = os.getenv('PUBLIC_URL_BASE', 'https://storage.googleapis.com')
INPUT_NOTEBOOK = os.getenv('INPUT_NOTEBOOK', '/app/notebooks/ProcessFITS.ipynb')

incoming_bucket = os.getenv('INPUT_BUCKET', 'panoptes-image-processing')
processed_bucket = os.getenv('OUTPUT_BUCKET', 'panoptes-processed-images')
error_bucket = os.getenv('ERROR_BUCKET', 'panoptes-images-error')


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
    bucket = message['bucket']
    if bucket == incoming_bucket:
        bucket_path = message['name']
        image_settings = ImageSettings(output_dir='temp', **from_json(message.get('imageSettings', '{}')))

        try:
            response = process_image(bucket_path, image_settings)
            response['success'] = True
        except Exception as e:
            print(f'Problem with processing from pubsub notification: {e}')

    return response


@app.post('/image/process/notebook')
def process_image(bucket_path, image_settings: ImageSettings, upload: bool = True):
    unit_doc_ref, seq_doc_ref, image_doc_ref = get_firestore_refs(bucket_path)

    try:
        image_status = image_doc_ref.get(['status']).to_dict()['status']
    except Exception:
        image_status = ImageStatus.UNKNOWN.name

    if ImageStatus[image_status] >= ImageStatus.PROCESSING:
        print(f'Skipping image with status of {image_status}')
        return dict(success=False, error=f'Skipping image with status of {image_status}')

    # Update the image status.
    print(f'Updating status for {bucket_path} to {ImageStatus.PROCESSING.name}')
    image_doc_ref.set({'status': ImageStatus.PROCESSING.name}, merge=True)

    # Assume we will upload to the processed bucket.
    upload_bucket = processed_bucket

    path_info = ImagePathInfo(path=bucket_path)
    with tempfile.TemporaryDirectory() as output_dir:
        image_settings.output_dir = output_dir

        incoming_image_path = f'/{incoming_bucket}/{bucket_path}'
        print(f'Processing {incoming_image_path} with {image_settings}')

        try:
            notebook_path, has_errors = process_notebook(incoming_image_path,
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
            upload_bucket = error_bucket
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
            try:
                ext = '.fits' if image_settings.compress_fits is False else '.fits.fz'
                outgoing_image_path = path_info.as_path(base=f'/{upload_bucket}', ext=ext)
                print(f'Moving {incoming_image_path} to {outgoing_image_path}')
                Path(incoming_image_path).rename(outgoing_image_path)
                return_dict['processed_bucket_path'] = outgoing_image_path
            except Exception as e3:
                print(f'Error moving {incoming_image_path} to {upload_bucket}: {e3!r}')
                return_dict['error_3'] = f'{e3!r}'

            # Copy any assets to the upload bucket.
            if upload:
                output_url_list = list()
                for fn in Path(output_dir).glob('*'):
                    print(f'Moving {fn} to {upload_bucket}')
                    new_path = Path(fn).rename(f'{upload_bucket}/{fn.name}')
                    output_url_list.append(new_path.as_posix())

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
