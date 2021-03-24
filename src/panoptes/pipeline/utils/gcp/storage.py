from google.cloud.storage import Bucket
from loguru import logger


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
