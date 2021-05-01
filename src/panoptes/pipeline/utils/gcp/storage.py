import os
import shutil
import subprocess

from google.cloud.storage import Bucket
from loguru import logger
from panoptes.utils.images import fits as fits_utils
from tqdm.auto import tqdm


def move_blob_to_bucket(blob_name: str, old_bucket: Bucket, new_bucket: Bucket,
                        remove: bool = True):
    """Copy and optionally remove the blob from old to new bucket.

    Args:
        blob_name (str): The relative path to the blob.
        old_bucket (Bucket): The name of the bucket where we get the file.
        new_bucket (Bucket): The name of the bucket where we move/copy the file.
        remove (bool, optional): If True (the default), file should be removed
            afterwards.
    """
    logger.info(f'Moving {blob_name} → {new_bucket}')
    old_bucket.copy_blob(old_bucket.get_blob(blob_name), new_bucket)
    if remove:
        old_bucket.delete_blob(blob_name)


def copy_blob_to_bucket(*args, **kwargs):
    """A thin-wrapper around `move_blob_to_bucket` that sets `remove=False`."""
    kwargs['remove'] = False
    move_blob_to_bucket(*args, **kwargs)


def download_images(image_list, output_dir, overwrite=False, unpack=True, show_progress=True):
    """Download images.

    Temporary helper script that needs to be more robust.
    """
    os.makedirs(output_dir, exist_ok=True)

    fits_files = list()

    iterator = image_list
    if show_progress:
        iterator = tqdm(iterator, desc='Downloading images')

    wget = shutil.which('wget')

    for fits_file in iterator:
        base = os.path.basename(fits_file)
        unpacked = base.replace('.fz', '')

        if not os.path.exists(f'{output_dir}/{base}') or overwrite:
            if not os.path.exists(f'{output_dir}/{unpacked}') or overwrite:
                download_cmd = [wget, '-q', fits_file, '-O', f'{output_dir}/{base}']
                subprocess.run(download_cmd)

        # Unpack the file if packed version exists locally.
        if os.path.exists(f'{output_dir}/{base}') and unpack:
            fits_utils.funpack(f'{output_dir}/{base}')

        if os.path.exists(f'{output_dir}/{unpacked}'):
            fits_files.append(f'{output_dir}/{unpacked}')

    logger.debug(f'Downloaded {len(fits_files)} files.')
    return fits_files
