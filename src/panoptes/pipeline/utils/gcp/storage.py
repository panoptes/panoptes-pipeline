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
            public_urls.append(blob.public_url)
        except ConnectionError as e:
            print(f'Error during upload of  {bucket_path}. {e!r}')

    return public_urls
