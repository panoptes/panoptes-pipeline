from pathlib import Path

import papermill as pm
from panoptes.data.images import ImagePathInfo

from panoptes.pipeline.settings import ImageSettings


def process_notebook(bucket_path: str,
                     input_notebook: Path,
                     output_dir: Path = Path('.'),
                     settings: ImageSettings = None
                     ) -> (str, bool):
    try:
        print(f'Checking if got a fits file at {bucket_path}')
        path_info = ImagePathInfo(path=bucket_path)
        print(f'Got image info: {path_info}')
    except ValueError as e:
        raise RuntimeError(f'Need a FITS file, got {bucket_path}')

    print(f'Starting image processing for {path_info} with {input_notebook} in {output_dir!r}')

    # Set proper names for the image settings.
    image_settings = settings if settings is not None else ImageSettings(
        output_dir=output_dir, files=dict(
            reduced_filename=f'{path_info.image_id}.fits.fz',
            sources_filename=f'{path_info.image_id}.sources.parquet'
        )
    )

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
                output_dir=str(image_settings.output_dir),
                image_settings=image_settings.model_dump_json()
            ),
            progress_bar=False,
            log_output=True
        )

    except Exception as e:
        has_errors = True
        print(f'Problem processing papermill notebook for {path_info}: {e!r}')
    else:
        # Upload the notebook to the processed bucket.
        pass

    finally:
        print(f'Finished processing {path_info} to {out_notebook}: {has_errors=}')
        return out_notebook, has_errors
