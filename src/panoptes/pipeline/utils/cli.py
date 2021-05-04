import typer

from panoptes.pipeline.utils.scripts.prepare import app as prepare_app

app = typer.Typer()
app.add_typer(prepare_app, name="prepare")

if __name__ == "__main__":
    app()
