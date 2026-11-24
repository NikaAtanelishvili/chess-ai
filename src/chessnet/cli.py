"""Command-line entry point (``chessnet``)."""

import logging
from pathlib import Path
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


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


@app.command()
def preprocess(
    config: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Data config (YAML), e.g. configs/data/lichess_2000plus.yaml.",
        ),
    ],
    source: Annotated[
        str,
        typer.Option(help="URL or local path of a Lichess PGN dump (.pgn.zst or .pgn)."),
    ],
    out: Annotated[
        Path,
        typer.Option(help="Output directory for the shards and manifest.json (must be empty)."),
    ],
    max_games: Annotated[
        int | None,
        typer.Option(min=1, help="Stop after this many games are kept (overrides the config)."),
    ] = None,
    workers: Annotated[
        int | None,
        typer.Option(min=0, help="Parser processes (default: CPU count; 0: in-process)."),
    ] = None,
) -> None:
    """Stream a PGN dump, filter its games and write shuffled training shards."""
    from chessnet.config import load_data_config
    from chessnet.data.pipeline import preprocess as run_preprocess

    _configure_logging()
    manifest = run_preprocess(
        load_data_config(config), source, out, max_games=max_games, workers=workers
    )
    for split, info in manifest.splits.items():
        shards = len(info.shards)
        typer.echo(f"{split}: {info.games:,} games, {info.positions:,} positions, {shards} shards")
