import os
import tempfile
from pathlib import Path

from fastapi import FastAPI

from google.cloud import storage

from panoptes.pipeline.utils.gcp.functions import cloud_function_entry_point
from panoptes.pipeline.utils.scripts.prepare import main as prepare_main

app = FastAPI()
storage_client = storage.Client()
output_bucket = storage_client.get_bucket(os.getenv('OUTPUT_BUCKET', 'panoptes-images-processed'))


@app.get('/')
async def root():
    return {'success': True}


@app.post('/prepare')
def index(message_envelope: dict,
          use_bigquery: bool = True,
          use_firestore: bool = True,
          force_new: bool = False):
    print(f'Received {message_envelope}')
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            full_image_id = cloud_function_entry_point(message_envelope['message'],
                                                       prepare_main,
                                                       output_dir=tmp_dir,
                                                       use_firestore=use_firestore,
                                                       use_bigquery=use_bigquery,
                                                       force_new=force_new,
                                                       )
        except Exception as e:
            print(f'Problem preparing an image: {e!r}')
            return {'success': False, 'error': f'{e!r}'}

        # Upload assets to storage bucket.
        for f in Path(tmp_dir).glob('*'):
            bucket_path = f'{full_image_id}/{f.name}'
            blob = output_bucket.blob(bucket_path)
            print(f'Uploading {bucket_path}')
            blob.upload_from_filename(f.absolute())

        # Success
        return {'success': True, 'location': f'gs://{output_bucket.name}/{full_image_id}'}
