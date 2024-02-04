from pathlib import Path
from typing import List

from google.cloud import storage


def upload_dir(directory: Path, prefix: str = '', bucket: storage.Bucket = None) -> List[str]:
    """Uploads all files in directory to storage bucket."""
    public_urls = list()
    for f in Path(directory).glob('*'):
        print(f'Uploading {prefix}/{f} to {bucket}')
        bucket_path = f'{prefix}/{f.name}'
        blob = bucket.blob(bucket_path)
        print(f'Uploading {bucket_path}')
        try:
            blob.upload_from_filename(str(f.absolute()))
            public_urls.append(blob.id)
        except ConnectionError as e:
            print(f'Error during upload of  {bucket_path}. {e!r}')

    return public_urls


def move_blob_to_bucket(blob_name: str,
                        old_bucket: str | storage.Bucket,
                        new_bucket: str | storage.Bucket,
                        remove: bool = True) -> storage.Blob:
    """Copy and optionally remove the blob from old to new bucket.

    Args:
        blob_name (str): The relative path to the blob.
        old_bucket (Bucket): The name of the bucket where we get the file.
        new_bucket (Bucket): The name of the bucket where we move/copy the file.
        remove (bool, optional): If True (the default), file should be removed
            afterward.
    """
    if isinstance(old_bucket, str):
        old_bucket = storage.Client().get_bucket(old_bucket)
    if isinstance(new_bucket, str):
        new_bucket = storage.Client().get_bucket(new_bucket)

    print(f'Moving {old_bucket=} {blob_name} → {new_bucket=}')
    new_blob = None
    try:
        new_blob = old_bucket.copy_blob(old_bucket.get_blob(blob_name), new_bucket)
        if remove:
            old_bucket.delete_blob(blob_name)
            print(f'Removed {blob_name} from {old_bucket}')
    except Exception as e:
        print(f'Error moving {blob_name} to {new_bucket} from {old_bucket}: {e!r}')

    return new_blob
