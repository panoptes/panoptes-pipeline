import os
import tempfile
from pathlib import Path
import re
from typing import Tuple, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl

from google.cloud import storage
from google.cloud import firestore
from google.cloud import pubsub

from panoptes.utils.images import fits as fits_utils
from panoptes.pipeline.utils.metadata import record_metadata, get_firestore_doc_ref, ImageStatus, ObservationPathInfo
from panoptes.pipeline.scripts.image import calibrate
from panoptes.pipeline.scripts.image import Settings as ImageSettings
from panoptes.pipeline.observation import get_stamp_locations, make_stamps
from panoptes.pipeline.utils.gcp.storage import move_blob_to_bucket

app = FastAPI()
storage_client = storage.Client()
firestore_db = firestore.Client()
publisher = pubsub.PublisherClient()

PROJECT_ID = os.getenv('PROJECT_ID', 'panoptes-exp')
EXTRACT_TOPIC = os.getenv('EXTRACT_STAMP_TOPIC', 'extract-stamps')
ROOT_URL = os.getenv('PUBLIC_URL_BASE', 'https://storage.googleapis.com')
processed_bucket = storage_client.get_bucket(os.getenv('OUTPUT_BUCKET', 'panoptes-images-processed'))
incoming_bucket = storage_client.get_bucket(os.getenv('INPUT_BUCKET', 'panoptes-images-incoming'))
error_bucket = storage_client.get_bucket(os.getenv('ERROR_BUCKET', 'panoptes-images-error'))
extract_stamp_topic = f'projects/{PROJECT_ID}/topics/{EXTRACT_TOPIC}'


class ObservationInfo(BaseModel):
    sequence_id: str
    frame_slice: Tuple[Optional[int], Optional[int]] = (1, None)
    stamp_size: Tuple[int, int] = (10, 10)
    base_url: HttpUrl = 'https://storage.googleapis.com/panoptes-images-processed'
    image_filename: Path = 'image.fits.fz'
    source_filename: Path = 'sources.parquet'
    image_status: ImageStatus = ImageStatus.MATCHED
    force_new: bool = False


@app.post('/image/process')
def process_image_from_storage(message_envelope: dict):
    print(f'Received {message_envelope}')

    message = message_envelope['message']
    bucket = message['attributes']['bucketId']
    bucket_path = message['attributes']['objectId']
    public_bucket_path = f'{ROOT_URL}/{bucket}/{bucket_path}'

    # Put things in the outgoing bucket unless errors below.
    asset_bucket = processed_bucket

    with tempfile.TemporaryDirectory() as tmp_dir:
        settings = ImageSettings(output_dir=tmp_dir)
        try:
            # Make sure file has valid signature, i.e. we need a FITS here.
            if re.search(r'\d{8}T\d{6}\.fits[.fz]+$', bucket_path) is None:
                raise RuntimeError(f'Need a FITS file, got {bucket_path}')

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
            metadata = calibrate(public_bucket_path, settings=settings)
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
                new_blob = move_blob_to_bucket(bucket_path, processed_bucket, error_bucket)
                return_dict['error_bucket_path'] = new_blob.path
            except Exception as e2:
                print(f'Error moving {bucket_path} to {error_bucket} from {incoming_bucket}: {e2!r}')
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


@app.post('/observation/get-stamp-locations')
def make_stamp_locations(observation_info: ObservationInfo):
    """Get the locations of the stamp and save as parquet file in bucket."""
    stamp_loc_bucket_path = f'{observation_info.sequence_id.replace("_", "/")}/stamp-positions.parquet'
    positions_bucket_path = processed_bucket.blob(stamp_loc_bucket_path)
    if positions_bucket_path.exists() and observation_info.force_new is False:
        return dict(success=False, message=f'Positions file already exists: {positions_bucket_path.public_url}')

    print(f'Getting stamp locations for {observation_info.sequence_id=}')
    unit_id, camera_id, sequence_time = observation_info.sequence_id.split('_')

    # Get sequence information
    sequence_doc_ref = firestore_db.document(f'units/{unit_id}/observations/{observation_info.sequence_id}')
    if sequence_doc_ref.get().exists is False and observation_info.force_new is False:
        return dict(success=False, message=f'No record for {observation_info.sequence_id}')

    # Get and show the metadata about the observation.
    matched_query = sequence_doc_ref.collection('images').where('status', '==', observation_info.image_status.name)
    matched_docs = [d.to_dict() for d in matched_query.stream()]
    images_df = pd.json_normalize(matched_docs, sep='_')

    # Set a time index.
    images_df.time = pd.to_datetime(images_df.time)
    images_df = images_df.set_index(['time']).sort_index()

    # Only use selected frames. TODO support time-based slicing(?)
    images_df = images_df[slice(*observation_info.frame_slice)]
    num_frames = len(images_df)
    print(f'Matched {num_frames} images for {observation_info.sequence_id=}')

    # Get the source files from the public url.
    sources_file_list = [f'{observation_info.base_url}/{i.replace("_", "/")}/{observation_info.source_filename}'
                         for i in images_df.uid.values]
    print(f'Loading {len(sources_file_list)} urls. Example: {sources_file_list[:1]}')

    try:
        stamp_positions = get_stamp_locations(sources_file_list=sources_file_list)
        print(f'Made {len(stamp_positions)} positions for {observation_info.sequence_id=}')

        # Save to storage bucket as parquet file.
        print(f'Saving stamp positions to {positions_bucket_path.name}')
        positions_bucket_path.upload_from_string(stamp_positions.to_parquet(), 'application/parquet')
        public_bucket_path = positions_bucket_path.public_url

        for full_id in images_df.uid.values:
            image_url = f'{observation_info.base_url}/{full_id.replace("_", "/")}/{observation_info.image_filename}'

            print(f'Sending pubsub message for {image_url}')
            publisher.publish(extract_stamp_topic, b'',
                              image_url=image_url,
                              positions_url=positions_bucket_path.public_url)
    except Exception as e:
        print(f'Error getting stamp positions: {e!r}')
    else:
        return dict(success=True, location=public_bucket_path)


@app.post('/image/extract-stamps')
def extract_stamps(message_envelope: dict):
    """Extract postage stamps from the given image.

    This function relies on a parquet file containing the positions for
    each source to be extracted. The positions file would usually be the
    same for an entire observation but that's not required.
    """
    print(f'Received pubsub message: {message_envelope!r}')
    message = message_envelope['message']
    image_url = message['attributes']['image_url']
    positions_url = message['attributes']['positions_url']

    # Get observation info and storage blob.
    path_info = ObservationPathInfo(path=image_url)
    stamp_url = f'{path_info.get_full_id(sep="/")}/stamps.parquet'
    stamps_blob = processed_bucket.blob(stamp_url)

    # Get the positions data.
    positions = pd.read_parquet(positions_url)
    print(f'Extracting stamps for {len(positions)} positions in {image_url}')

    # Get remote data
    data = fits_utils.getdata(image_url)
    stamps = make_stamps(positions, data)
    del data

    # Upload to bucket.
    stamps_blob.upload_from_string(stamps.to_parquet(), 'application/parquet')
    print(f'{len(stamps)} stamps uploaded to {stamps_blob.public_url}')

    # Update firestore record.
    doc_path = f'units/{path_info.unit_id}/observations/{path_info.sequence_id}/images/{path_info.image_id}'
    doc_updates = dict(status=ImageStatus.EXTRACTED.name, sources=dict(num_extracted=len(stamps)))
    firestore_db.document(doc_path).set(doc_updates, merge=True)


@app.post('/observation/make-observation-files')
def make_observation_files(observation_info: ObservationInfo):
    """Builds the PSC and metadata files for the entire observation."""
    print(f'Building files for {observation_info.sequence_id=}')
    unit_id, camera_id, sequence_time = observation_info.sequence_id.split('_')

    sequence_path = observation_info.sequence_id.replace("_", "/")
    psc_blob_name = f'{sequence_path}/stamp-collection.parquet'
    psc_blob_path = f'gcs://{processed_bucket.name}/{psc_blob_name}'
    psc_blob = processed_bucket.blob(psc_blob_name)

    if psc_blob.exists() and observation_info.force_new is False:
        return dict(success=False, message=f'PSC file already exists and force_new=False: {psc_blob.public_url}')

    # Get sequence information
    sequence_doc_ref = firestore_db.document(f'units/{unit_id}/observations/{observation_info.sequence_id}')
    if sequence_doc_ref.get().exists is False and observation_info.force_new is False:
        return dict(success=False, message=f'No record for {observation_info.sequence_id}')

    # Get the image ids that have had stamps extracted.
    matched_query = sequence_doc_ref.collection('images').where('status', '==', ImageStatus.EXTRACTED.name)
    matched_image_ids = [d.get('uid') for d in matched_query.stream()]

    # Build the storage location for each stamps file.
    stamps = list()
    for image_id in matched_image_ids:
        blob_path = f'gcs://{processed_bucket.name}/{image_id.replace("_", "/")}/stamps.parquet'
        print(f'Getting stamps for {image_id=} at {blob_path}')
        stamp_df = pd.read_parquet(blob_path)
        stamp_df['time'] = pd.to_datetime(image_id.split('_')[-1])
        stamp_df = stamp_df.set_index(['time'], append=True)
        stamps.append(stamp_df)

    # Create PSC file for entire observation.
    stamps_df = pd.concat(stamps).sort_index()
    print(f'Saving {len(stamps_df)} stamps for {observation_info.sequence_id} to {psc_blob_path}')
    stamps_df.to_parquet(psc_blob_path)
    del stamps_df

    sources_blob_name = f'{sequence_path}/source-collection.parquet'
    sources_blob_path = f'gcs://{processed_bucket.name}/{sources_blob_name}'
    sources_blob = processed_bucket.blob(sources_blob_name)

    # Build the joined metadata file.
    sources = list()
    for image_id in matched_image_ids:
        blob_path = f'gcs://{processed_bucket.name}/{image_id.replace("_", "/")}/sources.parquet'
        print(f'Getting sources for {image_id=} at {blob_path}')
        source_df = pd.read_parquet(blob_path)
        sources.append(source_df)

    # Create PSC file for entire observation.
    sources_df = pd.concat(sources).sort_index()

    # Merge the stamp positions.
    # positions_df = pd.read_parquet(f'gcs://{processed_bucket.name}/{sequence_path}/stamp-positions.parquet')
    # sources_df = sources_df.merge(positions_df, on='picid')

    print(f'Saving {len(sources_df)} sources for {observation_info.sequence_id} to {sources_blob_path}')
    sources_df.to_parquet(sources_blob_path)
    del sources_df

    return dict(success=True, psc_location=psc_blob.public_url, source_location=sources_blob.public_url)
