import os
import tempfile
from pathlib import Path
import re

from fastapi import FastAPI

from google.cloud import storage

from panoptes.pipeline.utils.gcp.functions import cloud_function_entry_point
from panoptes.pipeline.utils.scripts.prepare import main as prepare_main
from panoptes.pipeline.utils.gcp.storage import move_blob_to_bucket

app = FastAPI()
storage_client = storage.Client()
output_bucket = storage_client.get_bucket(os.getenv('OUTPUT_BUCKET', 'panoptes-images-processed'))
incoming_bucket = storage_client.get_bucket(os.getenv('INPUT_BUCKET', 'panoptes-images-incoming'))
error_bucket = storage_client.get_bucket(os.getenv('ERROR_BUCKET', 'panoptes-images-error'))


@app.get('/')
async def root():
    return {'success': True}


@app.post('/prepare')
def index(message_envelope: dict):
    print(f'Received {message_envelope}')

    message = message_envelope['message']
    bucket_path = message['attributes']['objectId']

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            # Make sure file has valid signature, i.e. we need a FITS here.
            if re.search(r'\d{8}T\d{6}\.fits[.fz]+$', bucket_path) is None:
                raise RuntimeError(f'Need a FITS file, got {bucket_path}')

            full_image_id = cloud_function_entry_point(message, prepare_main, output_dir=tmp_dir)
        except Exception as e:
            print(f'Problem preparing an image: {e!r}')

            # Move to error bucket.
            new_blob = move_blob_to_bucket(bucket_path, incoming_bucket, error_bucket)

            return {'success': False, 'error': f'{e!r}', 'error_bucket_path': new_blob.path}

        # Upload assets to storage bucket.
        for f in Path(tmp_dir).glob('*'):
            bucket_path = f'{full_image_id}/{f.name}'
            blob = output_bucket.blob(bucket_path)
            print(f'Uploading {bucket_path}')
            blob.upload_from_filename(f.absolute())

        # Success
        return {'success': True, 'location': f'gs://{output_bucket.name}/{full_image_id}'}
