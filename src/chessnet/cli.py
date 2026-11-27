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


WeightsOption = Annotated[
    Path | None,
    typer.Option(
        dir_okay=False,
        help="Network checkpoint (default: CHESSNET_WEIGHTS, else the chosen base network "
        "in checkpoints/session2/).",
    ),
]
DeviceOption = Annotated[str, typer.Option(help="'auto', 'cpu' or 'cuda'.")]


@app.command()
def uci(weights: WeightsOption = None, device: DeviceOption = "auto") -> None:
    """Run the UCI engine on stdin and stdout, for chess GUIs (same as chessnet-uci)."""
    from chessnet.engine.uci import main as serve

    serve(weights=str(weights) if weights else None, device=device)


@app.command()
def play(
    color: Annotated[
        str, typer.Option(help="Your colour: 'white', 'black' or 'random'.")
    ] = "white",
    weights: WeightsOption = None,
    device: DeviceOption = "auto",
    mode: Annotated[
        str, typer.Option(help="'mcts' (search) or 'policy' (instant, with safety rules).")
    ] = "mcts",
    simulations: Annotated[int, typer.Option(min=1, help="Playouts per engine move.")] = 800,
    movetime: Annotated[
        float | None, typer.Option(min=0.01, help="Seconds per engine move, instead of playouts.")
    ] = None,
) -> None:
    """Play a game against the engine in the terminal."""
    import random

    import chess

    from chessnet.engine.play import play_game
    from chessnet.engine.player import MCTSPlayer, Player, PolicySafePlayer
    from chessnet.engine.uci import default_weights, resolve_device, value_to_cp
    from chessnet.search.evaluator import NetworkEvaluator
    from chessnet.search.mcts import SearchConfig

    colours = {"white": chess.WHITE, "black": chess.BLACK}
    if color not in (*colours, "random"):
        raise typer.BadParameter("use 'white', 'black' or 'random'", param_hint="--color")
    if mode not in ("mcts", "policy"):
        raise typer.BadParameter("use 'mcts' or 'policy'", param_hint="--mode")
    path = weights or Path(default_weights())
    if not path.is_file():
        raise typer.BadParameter(f"no checkpoint at {path}", param_hint="--weights")
    human = colours[color] if color in colours else random.choice([chess.WHITE, chess.BLACK])
    evaluator = NetworkEvaluator.from_checkpoint(path, device=resolve_device(device))
    player: Player
    describe = None
    if mode == "policy":
        player = PolicySafePlayer(evaluator)
    else:
        search_player = MCTSPlayer(
            evaluator,
            simulations=None if movetime else simulations,
            movetime=movetime,
            config=SearchConfig(batch_size=32),
        )
        player = search_player

        def describe() -> str:
            result = search_player.last_result
            assert result is not None
            # The value is the engine's; show it for White, as GUIs do.
            white = result.move_value if human == chess.BLACK else -result.move_value
            return (
                f"eval {value_to_cp(white) / 100:+.2f} for White, {result.playouts} playouts "
                f"in {result.seconds:.1f} s"
            )

    play_game(player, human, describe=describe)


@app.command("export-onnx")
def export_onnx(
    weights: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Checkpoint to export.")
    ],
    out: Annotated[Path, typer.Option(help="ONNX file to write.")],
    opset: Annotated[int, typer.Option(min=17, help="ONNX opset.")] = 18,
) -> None:
    """Export a checkpoint's network to ONNX, for fast CPU inference (needs the onnx extra)."""
    from chessnet.model.checkpoint import load_network
    from chessnet.model.inference import export_onnx as run_export

    try:
        written = run_export(load_network(weights), out, opset=opset)
    except ModuleNotFoundError as error:
        raise typer.BadParameter(
            f"{error.name} is missing: install the onnx extra (uv sync --extra onnx)"
        ) from error
    typer.echo(f"wrote {written}")


@app.command("eval")
def evaluate(
    suite: Annotated[
        str, typer.Option(help="'quick' (about 10 minutes on a laptop CPU) or 'full'.")
    ] = "quick",
    weights: WeightsOption = None,
    compare: Annotated[
        Path | None,
        typer.Option(dir_okay=False, help="A second network to play against the first (full)."),
    ] = None,
    workers: Annotated[
        int, typer.Option(min=1, help="Games or puzzles in parallel, one process each.")
    ] = 1,
    device: DeviceOption = "auto",
    stockfish: Annotated[
        str | None, typer.Option(help="Stockfish binary (default: stockfish on PATH).")
    ] = None,
    puzzles: Annotated[
        Path | None,
        typer.Option(
            dir_okay=False,
            help="Puzzle CSV, or the Lichess puzzle database (.csv.zst) to sample 2,000 from in "
            "the full suite (default: the 500-puzzle sample in tests/fixtures).",
        ),
    ] = None,
    out: Annotated[Path, typer.Option(help="Directory for results JSON and PGNs.")] = Path(
        "eval/results"
    ),
    model_card: Annotated[Path, typer.Option(help="Model card for the summary table.")] = Path(
        "docs/MODEL_CARD.md"
    ),
    estimate: Annotated[
        bool,
        typer.Option(help="Time a few games and puzzles, estimate the suite's wall time, stop."),
    ] = False,
    from_results: Annotated[
        Path | None,
        typer.Option(
            exists=True, dir_okay=False, help="Only write the model card table from a results JSON."
        ),
    ] = None,
    only: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated parts of the suite (e.g. policy-safe-vs-random,puzzles-policy): "
            "run only these, and merge them into the suite's existing results.",
        ),
    ] = None,
    max_wall_time: Annotated[
        str | None,
        typer.Option(help="Stop the suite cleanly after this long, e.g. 6h; saved as incomplete."),
    ] = None,
    heartbeat: Annotated[
        float, typer.Option(min=1, help="Seconds between heartbeat lines during long matches.")
    ] = 60.0,
    games: Annotated[
        int | None,
        typer.Option(min=1, help="Games per match for the matches chosen with --only."),
    ] = None,
    label: Annotated[
        str | None,
        typer.Option(
            help="Record the matches chosen with --only as labelled variants, e.g. 'laptop CPU' "
            "(kept next to the planned matches)."
        ),
    ] = None,
    merge_results: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Merge a results JSON of this suite from another machine (PGNs next to it) "
            "into ours, then stop.",
        ),
    ] = None,
) -> None:
    """Evaluate a network: matches (Elo with 95% CI, PGNs saved) and Lichess puzzles."""
    import json
    import shutil

    from chessnet.engine.uci import default_weights
    from chessnet.eval.suite import (
        estimate_suite,
        merge_suite_results,
        plan_suite,
        run_suite,
        update_model_card,
    )
    from chessnet.eval.workers import Watchdog
    from chessnet.train.loop import parse_duration

    if from_results is not None:
        update_model_card(model_card, json.loads(from_results.read_text(encoding="utf-8")))
        typer.echo(f"updated {model_card}")
        return
    path = weights or Path(default_weights())
    if not path.is_file():
        raise typer.BadParameter(f"no checkpoint at {path}", param_hint="--weights")
    try:
        plan = plan_suite(
            suite,
            str(path),
            compare=str(compare) if compare else None,
            stockfish=stockfish or shutil.which("stockfish"),
            puzzles=puzzles,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--suite") from error
    for note in plan.notes:
        typer.echo(note)
    if estimate:
        estimate_suite(plan, workers=workers, device=device, report=typer.echo)
        return
    if merge_results is not None:
        try:
            merge_suite_results(
                plan, str(path), merge_results, out=out, model_card=model_card, report=typer.echo
            )
        except ValueError as error:
            raise typer.BadParameter(str(error), param_hint="--merge-results") from error
        return
    selected = {name.strip() for name in only.split(",")} if only else None
    if (games is not None or label) and selected is None:
        raise typer.BadParameter(
            "needs --only: it applies to the chosen matches", param_hint="--games/--label"
        )
    try:
        run_suite(
            plan, str(path), workers=workers, device=device, out=out, model_card=model_card,
            report=typer.echo, only=selected, watchdog=Watchdog(heartbeat=heartbeat),
            max_wall_time=parse_duration(max_wall_time) if max_wall_time else None,
            games=games, label=label,
        )  # fmt: skip
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--only") from error


def _numbers(text: str, option: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in text.split(",") if part.strip())
    except ValueError as error:
        raise typer.BadParameter(f"not a list of numbers: {text!r}", param_hint=option) from error
    if not values:
        raise typer.BadParameter("needs at least one number", param_hint=option)
    return values


@app.command()
def ladder(
    weights: WeightsOption = None,
    stockfish: Annotated[
        str | None, typer.Option(help="Stockfish binary (default: stockfish on PATH).")
    ] = None,
    levels: Annotated[
        str, typer.Option(help="Stockfish UCI_Elo levels to climb, lowest first.")
    ] = "1500,1700,1900,2100,2300",
    games: Annotated[int, typer.Option(min=2, help="Games per ladder level.")] = 40,
    stop_below: Annotated[
        float, typer.Option(min=0, max=1, help="Stop climbing after a level scored below this.")
    ] = 0.25,
    simulations: Annotated[int, typer.Option(min=1, help="Playouts per move on the ladder.")] = 800,
    scaling: Annotated[
        str, typer.Option(help="Playout budgets for the scaling curve ('' for none).")
    ] = "50,100,200,400,800,1600",
    scaling_games: Annotated[int, typer.Option(min=2, help="Games per scaling point.")] = 32,
    workers: Annotated[int, typer.Option(min=1, help="Games in parallel, one process each.")] = 1,
    device: DeviceOption = "auto",
    out: Annotated[Path, typer.Option(help="Directory for results JSON and PGNs.")] = Path(
        "eval/results"
    ),
    model_card: Annotated[Path, typer.Option(help="Model card for the tables.")] = Path(
        "docs/MODEL_CARD.md"
    ),
    max_wall_time: Annotated[
        str | None, typer.Option(help="Stop cleanly after this long, e.g. 3h30m.")
    ] = None,
    heartbeat: Annotated[
        float, typer.Option(min=1, help="Seconds between heartbeat lines during matches.")
    ] = 60.0,
    import_results: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Bring a ladder results JSON from another machine (PGNs next to it) here, "
            "write the chart and the model card section, then stop.",
        ),
    ] = None,
) -> None:
    """Rate MCTS on Stockfish's UCI_Elo scale (a ladder), and its strength against playouts."""
    import shutil

    from chessnet.engine.uci import default_weights
    from chessnet.eval.ladder import LadderPlan, import_ladder_results, run_ladder
    from chessnet.eval.workers import Watchdog
    from chessnet.train.loop import parse_duration

    if import_results is not None:
        try:
            import_ladder_results(import_results, out=out, model_card=model_card, report=typer.echo)
        except (ValueError, FileNotFoundError) as error:
            raise typer.BadParameter(str(error), param_hint="--import-results") from error
        typer.echo(f"updated {model_card}")
        return
    path = weights or Path(default_weights())
    if not path.is_file():
        raise typer.BadParameter(f"no checkpoint at {path}", param_hint="--weights")
    command = stockfish or shutil.which("stockfish")
    if command is None:
        raise typer.BadParameter("Stockfish was not found", param_hint="--stockfish")
    plan = LadderPlan(
        weights=str(path),
        stockfish=(command,),
        levels=tuple(sorted(_numbers(levels, "--levels"))),
        games=games,
        stop_below=stop_below,
        simulations=simulations,
        scaling=_numbers(scaling, "--scaling") if scaling.strip() else (),
        scaling_games=scaling_games,
    )
    run_ladder(
        plan, workers=workers, device=device, out=out, model_card=model_card, report=typer.echo,
        watchdog=Watchdog(heartbeat=heartbeat),
        max_wall_time=parse_duration(max_wall_time) if max_wall_time else None,
    )  # fmt: skip
    typer.echo(f"results: {out / (path.stem + '.ladder.json')}")
