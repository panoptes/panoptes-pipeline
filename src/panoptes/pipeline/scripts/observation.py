import os
import re
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Optional

import typer
import papermill as pm
from google.cloud import firestore, storage
from tqdm.auto import tqdm

from panoptes.pipeline.scripts.image import process_notebook as process_image_notebook
from panoptes.pipeline.utils.gcp.storage import upload_dir
from panoptes.pipeline.utils.metadata import ObservationStatus

app = typer.Typer()

OUTPUT_BUCKET = os.getenv('OUTPUT_BUCKET', 'panoptes-images-processed')
IMAGE_BUCKET = os.getenv('OUTPUT_BUCKET', 'panoptes-raw-images')
firestore_db = firestore.Client()
storage_client = storage.Client()

# Only want to match properly named file.
fits_matcher = re.compile(r'.*/\d{8}T\d{6}.fits.*?')


@app.command()
def process_notebook(sequence_id: str,
                     input_notebook: Path = 'ProcessObservation.ipynb',
                     output_dir: Optional[Path] = None,
                     process_images: bool = False,
                     upload: bool = False,
                     force_new: bool = False
                     ):
    """Process the observation."""
    typer.secho(f'Starting {sequence_id} processing')
    output_dir = output_dir or Path(tempfile.TemporaryDirectory().name)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_url_list = None

    processed_bucket = storage_client.get_bucket(OUTPUT_BUCKET)
    image_bucket = storage_client.get_bucket(IMAGE_BUCKET)

    sequence_path = sequence_id.replace('_', '/')
    unit_id, camera_id, sequence_time = sequence_id.split('_')

    # Check and update status.
    seq_ref = firestore_db.document(f'units/{unit_id}/observations/{sequence_id}')
    try:
        obs_status = seq_ref.get(['status']).to_dict()['status']
    except Exception:
        obs_status = ObservationStatus.UNKNOWN.name

    if ObservationStatus[obs_status] > ObservationStatus.CALIBRATED and force_new is False:
        typer.secho(f'Skipping: status={ObservationStatus[obs_status].name}', color='yellow')
        raise FileExistsError
    else:
        # Update status to show we're processing.
        seq_ref.set(dict(status=ObservationStatus.PROCESSING.name), merge=True)

    # Try to process all the images first.
    try:
        if process_images:
            # Get the FITS files in the image bucket.
            typer.secho(f'Getting FITS image files from {image_bucket}')
            fits_urls = [b.public_url
                         for b in
                         storage_client.get_bucket(image_bucket).list_blobs(
                             prefix=f'{sequence_path}/')
                         if fits_matcher.match(b.name)]

            for fits_url in tqdm(fits_urls):
                typer.secho(f'Processing image {fits_url}')
                with tempfile.TemporaryDirectory(
                        prefix=f'{str(output_dir.absolute())}/') as tmp_dir:
                    with suppress(FileExistsError):
                        process_image_notebook(fits_url, Path(tmp_dir), upload=upload)

        # Run process.
        out_notebook = f'{output_dir}/processing-observation.ipynb'
        typer.secho(f'Starting {input_notebook} processing')

        pm.execute_notebook(str(input_notebook),
                            str(out_notebook),
                            parameters=dict(
                                sequence_id=sequence_id,
                                output_dir=str(output_dir),
                            ),
                            progress_bar=False
                            )
    except Exception as e:
        typer.secho(f'Error processing notebook: {e!r}', color='yellow')
        seq_ref.set(dict(status=ObservationStatus.ERROR.name), merge=True)
    else:
        # Upload any assets to storage bucket.
        if upload:
            output_url_list = upload_dir(output_dir, prefix=f'{sequence_path}/',
                                         bucket=processed_bucket)

        seq_ref.set(dict(status=ObservationStatus.PROCESSED.name), merge=True)
    finally:
        typer.secho(f'Finished processing for {sequence_id=} in {output_dir!r}')
        return output_url_list


if __name__ == '__main__':
    app()
