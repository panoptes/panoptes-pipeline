import os
import tempfile
from contextlib import suppress
from pathlib import Path
import re

from fastapi import FastAPI

from google.cloud import storage
from google.cloud import firestore

from panoptes.pipeline.utils.metadata import record_metadata, get_firestore_doc_ref, ImageStatus
from panoptes.pipeline.scripts.image import process as process_image
from panoptes.pipeline.scripts.image import settings as image_settings
from panoptes.pipeline.utils.gcp.storage import move_blob_to_bucket

app = FastAPI()
storage_client = storage.Client()
firestore_db = firestore.Client()

ROOT_URL = os.getenv('PUBLIC_URL_BASE', 'https://storage.googleapis.com')
outgoing_bucket = storage_client.get_bucket(os.getenv('OUTPUT_BUCKET', 'panoptes-images-processed'))
incoming_bucket = storage_client.get_bucket(os.getenv('INPUT_BUCKET', 'panoptes-images-incoming'))
error_bucket = storage_client.get_bucket(os.getenv('ERROR_BUCKET', 'panoptes-images-error'))


@app.get('/')
async def root():
    return {'success': True}


@app.post('/process')
def index(message_envelope: dict):
    print(f'Received {message_envelope}')

    message = message_envelope['message']
    bucket = message['attributes']['bucketId']
    bucket_path = message['attributes']['objectId']
    public_bucket_path = f'{ROOT_URL}/{bucket}/{bucket_path}'

    # Put things in the outgoing bucket unless errors below.
    asset_bucket = outgoing_bucket

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            # Make sure file has valid signature, i.e. we need a FITS here.
            if re.search(r'\d{8}T\d{6}\.fits[.fz]+$', bucket_path) is None:
                raise RuntimeError(f'Need a FITS file, got {bucket_path}')

            image_settings.output_dir = Path(tmp_dir)

            # Check and update status.
            _, _, image_doc_ref = get_firestore_doc_ref(bucket_path, firestore_db=firestore_db)
            try:
                image_status = image_doc_ref.get(['status']).to_dict()['status']
            except Exception:
                image_status = ImageStatus.UNKNOWN.name

            if ImageStatus[image_status] > ImageStatus.CALIBRATING:
                print(f'Skipping image with status of {ImageStatus[image_status].name}')
                raise FileExistsError(f'Already processed {bucket_path}')
            else:
                image_doc_ref.set(dict(status=ImageStatus.CALIBRATING.name), merge=True)

            # Run process.
            metadata = process_image(public_bucket_path)
            full_image_id = metadata['image']['uid'].replace('_', '/')

            firestore_id = record_metadata(bucket_path, metadata, firestore_db=firestore_db)
            print(f'Recorded metadata in firestore id={firestore_id}')
        except FileExistsError as e:
            print(f'Skipping already processed file.')
            return_dict = {'success': False, 'error': f'{e!r}'}
        except Exception as e:
            print(f'Problem processing image for {bucket_path}: {e!r}')
            return_dict = {'success': False, 'error': f'{e!r}'}

            # Move to error bucket.
            try:
                new_blob = move_blob_to_bucket(bucket_path, outgoing_bucket, error_bucket)
                return_dict['error_bucket_path'] = new_blob.path
            except Exception as e2:
                print(f'Error moving {bucket_path} to {error_bucket} from {incoming_bucket}')
                return_dict['error_2'] = f'{e2!r}'
        else:
            return_dict = {'success': True, 'location': f'gs://{asset_bucket.name}/{full_image_id}'}

            # Upload any assets to storage bucket.
            for f in (Path(tmp_dir) / full_image_id).glob('*'):
                print(f'Uploading {f}')
                bucket_path = f'{full_image_id}/{f.name}'
                blob = asset_bucket.blob(bucket_path)
                print(f'Uploading {bucket_path}')
                blob.upload_from_filename(f.absolute())

        # Success.
        return return_dict
