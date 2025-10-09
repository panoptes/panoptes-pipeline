import os
from pathlib import Path

import papermill as pm
from google.cloud import firestore, storage
from panoptes.data.images import ImagePathInfo, ImageStatus
from panoptes.utils.serializers import from_json

from panoptes.pipeline.settings import ImageSettings
from panoptes.pipeline.utils.gcp.firestore import get_firestore_refs
from panoptes.pipeline.utils.gcp.storage import upload_dir
from panoptes.pipeline.utils.notebooks import convert_notebook

OUTPUT_BUCKET = os.getenv('OUTPUT_BUCKET', 'panoptes-images-processed')
IMAGE_BUCKET = os.getenv('INPUT_BUCKET', 'panoptes-images-incoming')
PROJECT_ID = os.getenv('PROJECT_ID', 'panoptes-exp')
firestore_db = firestore.Client()
storage_client = storage.Client()


def process_notebook(bucket_path: str,
                     input_notebook: Path = 'ProcessFITS.ipynb',
                     output_dir: Path = Path('.'),
                     image_settings: ImageSettings = ImageSettings(),
                     ) -> (str, bool):
    unit_doc_ref, seq_doc_ref, image_doc_ref = get_firestore_refs(bucket_path)
    force_process = image_settings.force_new

    try:
        image_dict = image_doc_ref.get(['status', 'forced_process']).to_dict()
        image_status = image_dict.get('status', ImageStatus.UNKNOWN.name)
        print(f'Current status for {bucket_path} is {ImageStatus[image_status].name}')
    except Exception:
        print(f'No status found for {bucket_path}, setting to {ImageStatus.UNKNOWN.name}')
        image_status = ImageStatus.UNKNOWN.name

    if force_process is False and ImageStatus[image_status] >= ImageStatus.PROCESSING:
        print(f'Skipping image with status of {image_status} and {force_process=}')
        return dict(success=False, error=f'Skipping image with status of {image_status} and {force_process=}')
    elif force_process is True and ImageStatus[image_status] == ImageStatus.PROCESSING:
        print(f'Image is currently being processed with status of {image_status} and {force_process=}')
        return dict(
            success=False,
            error=f'Skipping image currently being processed with status of {image_status} and {force_process=}'
        )

    # Update the image status.
    print(f'Updating status for {bucket_path} from {image_status} to {ImageStatus.PROCESSING.name}')
    image_doc_ref.set({'status': ImageStatus.PROCESSING.name}, merge=True)

    try:
        print(f'Checking if got a fits file at {bucket_path}')
        path_info = ImagePathInfo(path=bucket_path)
        print(f'Got image info: {path_info}')
    except ValueError as e:
        raise RuntimeError(f'Need a FITS file, got {bucket_path}')

    print(f'Starting image processing for {path_info} with {input_notebook} in {output_dir!r}')

    processed_bucket = storage_client.get_bucket(OUTPUT_BUCKET)

    # Run papermill process to execute the notebook.
    out_notebook = f'{output_dir}/{path_info.get_full_id()}-processed.ipynb'
    has_errors = False
    print(f'Running {input_notebook} to {out_notebook}')
    try:
        pm.execute_notebook(
            str(input_notebook),
            str(out_notebook),
            parameters=dict(
                fits_path=str(path_info.path),
                output_dir=output_dir.as_posix(),
                image_settings=image_settings.model_dump_json()
            ),
            progress_bar=False,
            log_output=True
        )

    except Exception as e:
        has_errors = True
        image_doc_ref.set(dict(status=ImageStatus.ERROR.name), merge=True)
        print(f'Problem processing papermill notebook for {path_info}: {e!r}')
    finally:
        # Convert the notebook to html
        convert_notebook(Path(input_notebook), Path(output_dir), Path(input_notebook).with_suffix('.html'))

        # Copy any assets to the upload bucket.
        output_url_list = list()
        fits_public_url = ''
        # Upload any assets to storage bucket.
        if image_settings.upload:
            output_url_list = upload_dir(output_dir, prefix=path_info.sequence_id, bucket=processed_bucket)

            if len(output_url_list) > 0:
                fits_public_url = list(filter(lambda a: 'fits' in a, output_url_list))
                fits_public_url = fits_public_url[0] if len(fits_public_url) else ''

        # If successful, write metadata to firestore and then remove the file.
        try:
            metadata_files = list(Path(f'{output_dir}/assets').glob('*metadata.json'))
            if len(metadata_files) == 0:
                raise FileNotFoundError(f'No metadata file found in {output_dir}!')
            metadata_file = metadata_files[0]
            if metadata_file.exists():
                with metadata_file.open() as f:
                    image_metadata = from_json(f.read())

                image_metadata['image']['fits_public_url'] = fits_public_url
                image_metadata['image']['assets'] = output_url_list
                image_metadata['image']['processed_time'] = firestore.SERVER_TIMESTAMP

                # unit_doc_ref.set(image_metadata['unit'], merge=True)
                seq_doc_ref.set(image_metadata['sequence'], merge=True)
                image_doc_ref.set(image_metadata['image'], merge=True)
                print(f'Recorded metadata for {bucket_path} with {image_doc_ref.id=}')

                # Remove the metadata file.
                # metadata_file.unlink()
            else:
                print(f'Metadata file not found at {metadata_file}')
        except FileNotFoundError:
            raise FileNotFoundError(f'No metadata file found in {output_dir}!')
        except Exception as e:
            print(f'Problem updating firestore with metadata: {e}')

        print(f'Finished processing {path_info} to {out_notebook}: {has_errors=}')
        return output_url_list
