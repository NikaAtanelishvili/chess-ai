"""Command-line entry point (``chessnet``)."""

from typing import Annotated

import typer

from chessnet import __version__

app = typer.Typer(
    name="chessnet",
    help="AlphaZero-style chess engine: preprocess, train, play and evaluate.",
    no_args_is_help=True,
    add_completion=False,
)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"chessnet {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_print_version,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """AlphaZero-style chess engine: preprocess, train, play and evaluate."""
