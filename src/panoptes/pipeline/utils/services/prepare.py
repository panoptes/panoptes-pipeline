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
outgoing_bucket = storage_client.get_bucket(os.getenv('OUTPUT_BUCKET', 'panoptes-images-processed'))
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

    # Put things in the outgoing bucket unless errors below.
    asset_bucket = outgoing_bucket

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            # Make sure file has valid signature, i.e. we need a FITS here.
            if re.search(r'\d{8}T\d{6}\.fits[.fz]+$', bucket_path) is None:
                raise RuntimeError(f'Need a FITS file, got {bucket_path}')

            full_image_id = cloud_function_entry_point(message, prepare_main, output_dir=tmp_dir)
        except Exception as e:
            print(f'Problem preparing an image for {bucket_path}: {e!r}')
            return_dict = {'success': False, 'error': f'{e!r}'}

            # Put assets in error bucket
            asset_bucket = error_bucket

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
        for f in Path(tmp_dir).glob('*'):
            bucket_path = f'{full_image_id}/{f.name}'
            blob = asset_bucket.blob(bucket_path)
            print(f'Uploading {bucket_path}')
            blob.upload_from_filename(f.absolute())

        # Success.
        return return_dict
