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


@app.command()
def train(
    config: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Training config (YAML), e.g. configs/train/kaggle_t4.yaml.",
        ),
    ],
    data: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Dataset written by chessnet preprocess."),
    ],
    out: Annotated[
        Path,
        typer.Option(help="Run directory for checkpoints and logs; reuse it to resume."),
    ],
    resume: Annotated[
        str,
        typer.Option(
            help="'auto' (newest checkpoint in --out or --resume-from), 'none', or a path."
        ),
    ] = "auto",
    resume_from: Annotated[
        list[Path] | None,
        typer.Option(help="Extra directories to search for checkpoints (e.g. a previous run)."),
    ] = None,
    max_wall_time: Annotated[
        str | None,
        typer.Option(help="Stop cleanly with a checkpoint after this long, e.g. 11h15m."),
    ] = None,
    max_steps: Annotated[
        int | None,
        typer.Option(min=1, help="Step budget for a new run (overrides the config)."),
    ] = None,
    logger: Annotated[
        list[str] | None,
        typer.Option(help="Extra metric logger: 'wandb' (needs WANDB_API_KEY)."),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option(help="'cuda' or 'cpu' (default: cuda if available)."),
    ] = None,
) -> None:
    """Train the network on a preprocessed dataset, resuming from the newest checkpoint."""
    from chessnet.config import load_train_config
    from chessnet.train.loop import parse_duration
    from chessnet.train.loop import train as run_training

    loggers = set(logger or [])
    if loggers - {"wandb"}:
        raise typer.BadParameter(
            f"unknown logger(s): {', '.join(sorted(loggers - {'wandb'}))}; console, CSV and "
            "TensorBoard are always on, and 'wandb' is the only optional one",
            param_hint="--logger",
        )
    train_config = load_train_config(config)
    if max_steps is not None:
        train_config = train_config.model_copy(update={"max_steps": max_steps})
    _configure_logging()
    result = run_training(
        train_config,
        data,
        out,
        resume=resume,
        resume_dirs=resume_from or [],
        max_wall_time=parse_duration(max_wall_time) if max_wall_time else None,
        wandb="wandb" in loggers,
        device=device,
    )
    typer.echo(
        f"stopped at step {result.state.step:,} of {result.state.total_steps:,} "
        f"({result.stopped_because}); last checkpoint: {result.last_checkpoint}"
    )
