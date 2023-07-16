import sys
from pathlib import Path
from typing import Optional

import typer

from animatediff import __version__, console, err_console

app = typer.Typer()

def version_callback(value: bool):
    if value:
        console.print(f"{__package__} v{__version__}")
        raise typer.Exit()


@app.callback()
def callback(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True, help="Show version"
    ),
):
    pass

@app.command()
def cli(
    verbose: bool = typer.Option(
        False, "-v", "--verbose", is_flag=True, help="Enable verbose console output."
    ),
    option: Path = typer.Option(
        "./option.ini", "-o", "--option", exists=True, dir_okay=False, help="Path to a file"
    ),
    argument: str = typer.Argument(..., help="An argument"),
):
    """
    Main entrypoint for your application.
    """
    console.log(f"verbose: {verbose}")
    console.log(f"option: {option}")
    console.log(f"argument: {argument}")
    console.log("done")
    sys.exit(0)
