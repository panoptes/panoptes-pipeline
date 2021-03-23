from google.cloud.storage import Bucket


def move_blob_to_bucket(blob_name: str, old_bucket: Bucket, new_bucket: Bucket,
                        remove: bool = True):
    """Copy the blob from the incoming bucket to the `new_bucket`.

    Args:
        blob_name (str): The relative path to the blob.
        old_bucket (Bucket): The name of the bucket where we get the file.
        new_bucket (Bucket): The name of the bucket where we move/copy the file.
        remove (bool, optional): If file should be removed afterwards, i.e. a move, or just copied.
            Default True as per the function name.
    """
    print(f'Moving {blob_name} → {new_bucket}')
    old_bucket.copy_blob(old_bucket.get_blob(blob_name), new_bucket)
    if remove:
        old_bucket.delete_blob(blob_name)


def copy_blob_to_bucket(*args, **kwargs):
    kwargs['remove'] = False
    move_blob_to_bucket(*args, **kwargs)
