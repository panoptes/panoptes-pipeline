import os
import tempfile
from pathlib import Path
from typing import Tuple, Optional
from urllib.error import HTTPError

import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
from fastapi import FastAPI
from google.cloud import firestore
from google.cloud import pubsub
from google.cloud import storage
from panoptes.utils.images import fits as fits_utils
from panoptes.utils.serializers import from_json
from pydantic import BaseModel, HttpUrl

from panoptes.pipeline.observation import get_stamp_locations, make_stamps
from panoptes.pipeline.scripts.image import Settings as ImageSettings, calibrate
from panoptes.pipeline.utils.gcp.storage import move_blob_to_bucket
from panoptes.pipeline.utils.metadata import ImageStatus, \
    ObservationPathInfo, ObservationStatus

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
def process_image_from_storage(message_envelope: dict):
    print(f'Received {message_envelope}')

    message = message_envelope['message']
    bucket_path = message['attributes']['objectId']
    image_settings = from_json(message['attributes'].get('imageSettings', '{}'))

    with tempfile.TemporaryDirectory() as tmp_dir:
        image_settings['output_dir'] = tmp_dir
        image_settings = ImageSettings(**image_settings)
        try:
            calibrate(bucket_path, image_settings, firestore_db=firestore_db)
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


@app.post('/observation/get-stamp-locations')
def make_stamp_locations(metadata: ObservationInfo):
    """Get the locations of the stamp and save as parquet file in bucket."""
    stamp_loc_bucket_path = f'{metadata.sequence_id.replace("_", "/")}/stamp-positions.parquet'
    positions_bucket_path = processed_bucket.blob(stamp_loc_bucket_path)
    if positions_bucket_path.exists() and metadata.force_new is False:
        return dict(success=False,
                    message=f'Positions file already exists: {positions_bucket_path.public_url}')

    print(f'Getting stamp locations for {metadata.sequence_id=}')
    unit_id, camera_id, sequence_time = metadata.sequence_id.split('_')

    # Get sequence information
    sequence_doc_path = f'units/{unit_id}/observations/{metadata.sequence_id}'
    sequence_doc_ref = firestore_db.document(sequence_doc_path)
    if sequence_doc_ref.get().exists is False:
        return dict(success=False, message=f'No record for {metadata.sequence_id}')

    # Get and show the metadata about the observation.
    matched_query = sequence_doc_ref.collection('images').where('status', '==',
                                                                metadata.image_status.name)
    matched_docs = [d.to_dict() for d in matched_query.stream()]
    images_df = pd.json_normalize(matched_docs, sep='_')

    # Set a time index.
    images_df.time = pd.to_datetime(images_df.time)
    images_df = images_df.set_index(['time']).sort_index()

    # Only use selected frames. TODO support time-based slicing(?)
    images_df = images_df[slice(*metadata.frame_slice)]
    num_frames = len(images_df)

    # Sigma filtering of certain stats
    mask_columns = [
        'camera_colortemp',
        'sources_num_detected',
        'sources_photutils_fwhm_mean'
    ]
    frame_mask = np.zeros_like(images_df.index, dtype=bool)
    for mask_col in mask_columns:
        col_mask = sigma_clip(images_df[mask_col]).mask
        if col_mask.any():
            print(f'{mask_col} has {col_mask.sum()} masked frames')
            frame_mask = np.logical_or(frame_mask, col_mask)

    # Mark the masked frames in the metadata.
    for full_image_id in images_df.iloc[frame_mask].uid.to_list():
        unit_id, camera_id, sequence_time, image_time = full_image_id.split('_')
        image_doc_path = f'{sequence_doc_path}/images/{unit_id}_{camera_id}_{image_time}'
        print(f'Setting status={ImageStatus.MASKED.name} for {image_doc_path}')
        firestore_db.document(image_doc_path).set(dict(status=ImageStatus.MASKED.name), merge=True)

    # Get non-masked frames. We select the inverse of the mask to get proper unmasked items.
    images_df = images_df.iloc[~frame_mask]
    print(f'Matched {num_frames} images for {metadata.sequence_id=}')

    # Get the source files from the public url.
    sources_file_list = [f'{metadata.base_url}/{i.replace("_", "/")}/{metadata.source_filename}'
                         for i in images_df.uid.values]
    print(f'Loading {len(sources_file_list)} urls. Example: {sources_file_list[:1]}')

    try:
        stamp_positions, stamp_size = get_stamp_locations(sources_file_list=sources_file_list)
        print(f'Made {len(stamp_positions)} positions for {metadata.sequence_id=}')

        # Save to storage bucket as parquet file.
        print(f'Saving stamp positions to {positions_bucket_path.name}')
        positions_bucket_path.upload_from_string(stamp_positions.to_parquet(),
                                                 'application/parquet')
        public_bucket_path = positions_bucket_path.public_url

        for full_id in images_df.uid.values:
            image_url = f'{metadata.base_url}/{full_id.replace("_", "/")}/{metadata.image_filename}'

            print(f'Sending pubsub message for {image_url}')
            publisher.publish(extract_stamp_topic, b'',
                              image_url=image_url,
                              positions_url=positions_bucket_path.public_url)
    except Exception as e:
        print(f'Error getting stamp positions: {e!r}')
        sequence_doc_ref.set(dict(status=ObservationStatus.ERROR.name), merge=True)
    else:
        sequence_doc_ref.set(dict(status=ObservationStatus.MATCHED.name), merge=True)
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
    sequence_doc_path = f'units/{path_info.unit_id}/observations/{path_info.sequence_id}'
    image_doc_path = f'{sequence_doc_path}/images/{path_info.image_id}'

    # Get the positions data.
    positions = pd.read_parquet(positions_url)
    print(f'Extracting stamps for {len(positions)} positions in {image_url}')

    # Get remote data and process if available.
    try:
        data = fits_utils.getdata(image_url)
    except HTTPError as e:
        print(f'Error loading {image_url} {e!r}')
        firestore_db.document(image_doc_path).set(dict(status=ImageStatus.UNKNOWN.name), merge=True)
        return dict(success=False)

    stamps = make_stamps(positions, data)
    del data

    # Upload to bucket.
    stamps_blob.upload_from_string(stamps.to_parquet(), 'application/parquet')
    print(f'{len(stamps)} stamps uploaded to {stamps_blob.public_url}')

    # Update firestore record.
    doc_updates = dict(status=ImageStatus.EXTRACTED.name, sources=dict(num_extracted=len(stamps)))
    firestore_db.document(image_doc_path).set(doc_updates, merge=True)


@app.post('/observation/make-observation-files')
def make_observation_files(metadata: ObservationInfo):
    """Builds the PSC and metadata files for the entire observation."""
    print(f'Building files for {metadata.sequence_id=}')
    unit_id, camera_id, sequence_time = metadata.sequence_id.split('_')

    sequence_path = metadata.sequence_id.replace("_", "/")
    psc_blob_name = f'{sequence_path}/stamp-collection.parquet'
    psc_blob_path = f'gcs://{processed_bucket.name}/{psc_blob_name}'
    psc_blob = processed_bucket.blob(psc_blob_name)

    if psc_blob.exists() and metadata.force_new is False:
        return dict(success=False,
                    message=f'PSC file already exists and force_new=False: {psc_blob.public_url}')

    # Get sequence information
    sequence_doc_ref = firestore_db.document(f'units/{unit_id}/observations/{metadata.sequence_id}')
    if sequence_doc_ref.get().exists is False and metadata.force_new is False:
        return dict(success=False, message=f'No record for {metadata.sequence_id}')

    # Get the image ids that have had stamps extracted.
    matched_query = sequence_doc_ref.collection('images').where('status', '==',
                                                                ImageStatus.EXTRACTED.name)
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
    print(f'Gathered {len(stamps)} stamp files, making PSC.')
    stamps_df = pd.concat(stamps).sort_index()
    del stamps
    print(f'Saving PSC with {len(stamps_df)} stamps for {metadata.sequence_id} to {psc_blob_path}')
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
    print(f'Merging source files.')
    sources_df = pd.concat(sources).sort_index()
    del sources

    # Merge the stamp positions.
    positions_blob_name = f'{sequence_path}/stamp-positions.parquet'
    positions_blob_path = f'gcs://{processed_bucket.name}/{positions_blob_name}'
    positions_df = pd.read_parquet(positions_blob_path)
    sources_df = sources_df.reset_index().merge(positions_df, on='picid').set_index(
        ['picid', 'time']).sort_index()

    print(f'Saving {len(sources_df)} sources for {metadata.sequence_id} to {sources_blob_path}')
    sources_df.to_parquet(sources_blob_path)
    del positions_df
    del sources_df

    # Update status of observation.
    sequence_doc_ref.set(dict(status=ObservationStatus.PROCESSED.name), merge=True)

    return dict(success=True, psc_location=psc_blob.public_url,
                source_location=sources_blob.public_url)
