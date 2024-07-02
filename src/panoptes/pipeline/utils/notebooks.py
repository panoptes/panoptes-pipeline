import traceback
from pathlib import Path

from nbconvert import HTMLExporter


def convert_notebook(out_notebook, output_dir, output_notebook) -> Path:
    """ Convert a notebook to HTML.

    Args:
        out_notebook (Path): The path to the notebook.
        output_dir (Path): The directory to save the HTML.
        output_notebook (Path): The name of the output notebook.

    Returns:
        Path: The path to the converted notebook.

    Raises:
        Exception: If there is an error converting the notebook.
    """
    try:
        html_exporter = HTMLExporter()
        html_exporter.exclude_input = True
        html_exporter.exclude_output_prompt = True
        html_exporter.exclude_input_prompt = True
        html_body, resources = html_exporter.from_filename(out_notebook.as_posix())
        converted_notebook = Path(output_dir / output_notebook.with_suffix('.html'))
        converted_notebook.write_text(html_body)
        return converted_notebook
    except Exception as e:
        print(f'Error converting notebook to HTML: {e!r}')
        print(traceback.format_exc())
