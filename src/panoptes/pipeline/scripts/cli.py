import typer

from panoptes.pipeline.scripts.image import app as image_app

app = typer.Typer()
app.add_typer(image_app, name="image")

if __name__ == "__main__":
    app()
