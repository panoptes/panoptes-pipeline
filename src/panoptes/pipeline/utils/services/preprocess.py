import os
import tempfile
from pathlib import Path

from fastapi import FastAPI

from google.cloud import storage

from panoptes.pipeline.utils.gcp.functions import cloud_function_entry_point
from panoptes.pipeline.utils.scripts.preprocess import main as preprocess_main

app = FastAPI()
storage_client = storage.Client()
output_bucket = storage_client.get_bucket(os.getenv('OUTPUT_BUCKET', 'panoptes-images-processed'))


@app.get('/')
async def root():
    return {'success': True}, 200


@app.post('/prepare')
def index(raw_message: dict):
    print(f'Received {raw_message}')
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            sequence_id = cloud_function_entry_point(raw_message, preprocess_main,
                                                     output_dir=tmp_dir)
        except Exception as e:
            print(f'Problem preparing an image: {e!r}')
            return {'success': False, 'error': f'{e!r}'}, 204

        # Upload assets to storage bucket.
        for f in Path(tmp_dir).glob('*'):
            bucket_path = f'{sequence_id}/{f.name}'
            blob = output_bucket.blob(bucket_path)
            print(f'Uploading {bucket_path}')
            blob.upload_from_filename(f.absolute())

        # Success
        return {'success': True, 'location': f'{output_bucket.name}/{sequence_id}'}, 204
