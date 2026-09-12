#!/usr/bin/env python

import json
import os
from pathlib import Path

from panoptes.data.observations import ObservationInfo
from papermill import execute_notebook
from tqdm.contrib.concurrent import process_map
from typer import Typer

app = Typer()

os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'


@app.command(name='process-fits')
def main(directory: str = None, sequence_id: str = None, picid: str = None):
    """Get the observation info and start processing in parallel."""
    if directory is not None:
        print(f'Using directory {directory}')
        sequence_id = '_'.join(Path(directory).absolute().as_posix().split('/')[-3:])

    if sequence_id is None:
        raise ValueError('Must provide either a directory or sequence_id')

    print(f'Getting params for {sequence_id=}')
    params = get_params(sequence_id=sequence_id, path=directory)
    print(f'Processing {len(params)} notebooks for {sequence_id=}')
    process_map(do_image_notebook, params, max_workers=5)

    print(f'Processing observation notebook')
    do_observation_notebook(sequence_id)

    if picid is not None:
        print(f'Processing lightcurve notebook for {picid=}')
        do_lightcurve_notebook(sequence_id, picid)


def get_params(sequence_id: str, path: str | Path | None = None):
    input_notebook = Path('notebooks/ProcessFits.ipynb')
    output_dir = Path(f'notebooks/output/') / sequence_id

    if path is not None:
        # Get all the images in the path.
        print(f'Getting images from {path}')
        image_list = list(Path(path).glob('*.fz'))
    else:
        print(f'Getting observation info for {sequence_id}')
        obs_info = ObservationInfo(sequence_id=sequence_id)
        image_list = obs_info.image_list

    print(f'Found {len(image_list)} images')

    param_list = list()
    for url in image_list:
        # Check if url is a real url or path; if path, convert to posix string.
        if isinstance(url, Path):
            url = Path(url).absolute().as_posix()

        image_id = url.split('/')[-1].split('.')[0]
        file_paths = dict(
            files=dict(
                sources_filename=f'sources-{image_id}.parquet',
                reduced_filename=f'image-{image_id}.fits',
                metadata_filename=f'metadata-{image_id}.json',
            )
        )

        output_notebook = output_dir / f'ProcessFits-{image_id}.ipynb'

        param_list.append(
            dict(
                input_path=input_notebook.absolute().as_posix(),
                output_path=output_notebook.absolute().as_posix(),
                parameters=dict(
                    fits_path=url,
                    output_dir=output_dir.absolute().as_posix(),
                    image_settings=json.dumps(file_paths),
                )
            )
        )

    return param_list


def do_image_notebook(input_params):
    input_path = input_params.pop('input_path')
    output_path = input_params.pop('output_path')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    parameters = input_params.pop('parameters')
    if Path(output_path).exists():
        print(f'Notebook {output_path} already exists, skipping')
        return

    try:
        execute_notebook(input_path, output_path, parameters, progress_bar=False)
    except Exception as e:
        print(f'Error processing {input_path} to {output_path} {e!r}')


def do_observation_notebook(seq_id):
    input_notebook = Path('notebooks/ProcessObservation.ipynb')
    output_dir = Path(f'notebooks/output/') / seq_id
    output_notebook = output_dir / input_notebook.name

    if output_notebook.exists():
        print(f'Observation notebook {output_notebook} already exists, skipping')
        return

    execute_notebook(
        input_notebook.as_posix(), output_notebook.absolute().as_posix(), dict(
            sequence_id=seq_id,
            output_dir=output_dir.absolute().as_posix(),
        ), progress_bar=True
        )


def do_lightcurve_notebook(seq_id, picid):
    input_notebook = Path('notebooks/MakeLightcurves.ipynb')
    output_dir = Path(f'notebooks/output/') / seq_id
    output_notebook = output_dir / f'{picid}-{input_notebook.name}'

    if output_notebook.exists():
        print(f'Lightcurve notebook {output_notebook} already exists, skipping')
        return

    execute_notebook(
        input_notebook.absolute().as_posix(), output_notebook.absolute().as_posix(), dict(
            sequence_id=seq_id,
            output_dir=output_dir.absolute().as_posix(),
            picid=str(picid),
        ), progress_bar=True
        )


if __name__ == '__main__':
    app()
