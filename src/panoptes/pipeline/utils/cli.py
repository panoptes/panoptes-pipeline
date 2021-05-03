import typer

from panoptes.pipeline.utils.scripts.preprocess import app as preprocess_app

app = typer.Typer()
app.add_typer(preprocess_app, name="prepare")

if __name__ == "__main__":
    app()
