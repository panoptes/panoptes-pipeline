import typer

from panoptes.pipeline.scripts.observation import app as observation_app

app = typer.Typer()
app.add_typer(observation_app, name="observation")

if __name__ == "__main__":
    app()
