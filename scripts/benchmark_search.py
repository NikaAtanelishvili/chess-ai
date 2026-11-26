"""Search throughput: MCTS nodes (playouts) per second with a network, per leaf batch size.

Runs a fresh search of ``--simulations`` playouts from each of a few positions (opening,
middlegame and endgame, both sides to move; each board keeps its move history) and reports
nodes per second, network positions per second and how full the leaf batches were.

    python scripts/benchmark_search.py --checkpoint checkpoints/session2/base-best.pt
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import chess
import torch

from chessnet.config import MODEL_PRESETS
from chessnet.model.resnet import ChessNet
from chessnet.search.evaluator import NetworkEvaluator
from chessnet.search.mcts import MCTS, SearchConfig

#: Positions as moves from the start position (so repetitions see a real game), or a FEN.
POSITIONS = {
    "start position": "",
    "Ruy Lopez (Black to move)": "e4 e5 Nf3 Nc6 Bb5",
    "Queen's Gambit Declined (Black to move)": "d4 d5 c4 e6 Nc3 Nf6 Bg5",
    "Italian middlegame": "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d4 exd4 cxd4 Bb4+ Bd2 Bxd2+ Nbxd2 d5",
    "Lucena rook endgame": "fen 1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1",
}


def board_for(position: str) -> chess.Board:
    if position.startswith("fen "):
        return chess.Board(position.removeprefix("fen "))
    board = chess.Board()
    for san in position.split():
        board.push_san(san)
    return board


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="a training checkpoint")
    source.add_argument("--model", choices=MODEL_PRESETS, help="random weights of a preset")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--simulations", type=int, default=800, help="playouts per search")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 16, 32])
    parser.add_argument("--out", type=Path, help="also write the report to this Markdown file")
    parser.add_argument("--json", type=Path, help="also write the measurements to this file")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.checkpoint:
        evaluator = NetworkEvaluator.from_checkpoint(args.checkpoint, device=device)
        network_name = str(args.checkpoint)
    else:
        evaluator = NetworkEvaluator(ChessNet(MODEL_PRESETS[args.model]), device=device)
        network_name = f"{args.model} (random weights)"
    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else cpu_name()
    boards = {name: board_for(position) for name, position in POSITIONS.items()}

    # Warm-up: first calls initialise the device, cuDNN and the allocator.
    MCTS(evaluator, SearchConfig(batch_size=max(args.batch_sizes))).search(
        chess.Board(), simulations=min(args.simulations, 200)
    )
    rows: list[dict[str, object]] = []
    for batch_size in args.batch_sizes:
        for name, board in boards.items():
            evaluator.clear_cache()
            positions, batches = evaluator.positions, evaluator.batches
            started = time.perf_counter()
            result = MCTS(evaluator, SearchConfig(batch_size=batch_size)).search(
                board, simulations=args.simulations
            )
            seconds = time.perf_counter() - started
            evaluated = evaluator.positions - positions
            calls = max(evaluator.batches - batches, 1)
            rows.append(
                {
                    "batch_size": batch_size,
                    "position": name,
                    "nodes_per_second": result.playouts / seconds,
                    "network_positions_per_second": evaluated / seconds,
                    "mean_batch": evaluated / calls,
                    "best_move": board.san(result.move),
                    "value": result.value,
                }
            )

    lines = [
        f"# Search throughput ({hardware}, {device.type}, torch {torch.__version__}, "
        f"{torch.get_num_threads()} CPU threads)",
        "",
        f"Network: {network_name}. Each search: {args.simulations} playouts from a fresh tree "
        "and an empty cache. Nodes are playouts; network positions exclude cache hits and "
        "terminal positions.",
        "",
        "| Leaf batch | Nodes/s (mean) | Nodes/s (min-max) | Network positions/s | Mean batch |",
        "|---:|---:|---:|---:|---:|",
    ]
    summary = []
    for batch_size in args.batch_sizes:
        mine = [row for row in rows if row["batch_size"] == batch_size]
        speeds = [float(row["nodes_per_second"]) for row in mine]  # type: ignore[arg-type]
        network = statistics.mean(float(row["network_positions_per_second"]) for row in mine)  # type: ignore[arg-type]
        fill = statistics.mean(float(row["mean_batch"]) for row in mine)  # type: ignore[arg-type]
        summary.append({"batch_size": batch_size, "nodes_per_second": statistics.mean(speeds)})
        lines.append(
            f"| {batch_size} | {statistics.mean(speeds):,.0f} | {min(speeds):,.0f}-"
            f"{max(speeds):,.0f} | {network:,.0f} | {fill:.1f} |"
        )
    lines += ["", "Per position (largest batch):", ""]
    for row in rows:
        if row["batch_size"] == max(args.batch_sizes):
            lines.append(
                f"- {row['position']}: {float(row['nodes_per_second']):,.0f} nodes/s, best "  # type: ignore[arg-type]
                f"{row['best_move']}, value {float(row['value']):+.2f}"  # type: ignore[arg-type]
            )
    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
    if args.json:
        document = {"hardware": hardware, "device": device.type, "network": network_name}
        payload = {**document, "simulations": args.simulations, "summary": summary, "rows": rows}
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
