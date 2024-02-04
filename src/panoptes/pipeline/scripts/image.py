from pathlib import Path
import warnings
import typer

from panoptes.pipeline.image import process_notebook

warnings.simplefilter(action='ignore', category=FutureWarning)

app = typer.Typer()


@app.command(name='process-notebook')
def process_image_notebook(fits_path: str,
                           output_dir: Path,
                           input_notebook: Path = 'ProcessFITS.ipynb',
                           ):
    print(f'Starting image processing for {fits_path} in {output_dir!r}')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_notebook = None
    try:
        # Run papermill process to execute the notebook.
        output_notebook, has_errors = process_notebook(fits_path, input_notebook, output_dir=output_dir)
    except Exception as e:
        print(f'Error processing {fits_path}: {e!r}')

    return output_notebook


if __name__ == '__main__':
    app()
