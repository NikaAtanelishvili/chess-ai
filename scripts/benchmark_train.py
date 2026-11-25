"""Training throughput benchmark: positions per second per model, batch size and speed option.

Times complete training steps (GPU unpacking, forward, backward, optimizer step) with fp16
mixed precision on CUDA, for each combination of ``torch.compile`` and channels_last (both
off and on by default). The network is prepared with the training loop's own helpers, so
the options mean the same as in ``chessnet train``. It also times forward passes alone
(what validation costs) and, with ``--data``, the data loader, and turns the measured speed
into hours per epoch for that dataset.

    python scripts/benchmark_train.py --data /kaggle/input/.../shards --out benchmark.md
"""

from __future__ import annotations

import argparse
import gc
import itertools
import math
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from chessnet.config import MODEL_PRESETS, TrainConfig
from chessnet.data.dataset import ShardDataset, make_loader
from chessnet.data.shards import Split, load_shard, read_manifest
from chessnet.encoding.packed import (
    PackedRecords,
    batch_from_records,
    empty_records,
    unpack_batch_torch,
)
from chessnet.model.resnet import ChessNet, count_parameters
from chessnet.train.loop import make_optimizer, model_input, prepare_network
from chessnet.train.losses import compute_losses

SWITCH = {"off": False, "on": True}


@dataclass(frozen=True)
class Result:
    model: str
    batch_size: int
    channels_last: bool
    compile: bool
    train_speed: float  # positions/s
    eval_speed: float  # positions/s of forward passes; NaN when compiled (validation is eager)
    first_step: float  # seconds, compilation included
    memory: float  # peak GB; NaN on the CPU


def random_records(count: int, seed: int = 0) -> PackedRecords:
    rng = np.random.default_rng(seed)
    records = empty_records(count)
    records["bb"] = rng.integers(0, 2**63, size=(count, 12), dtype=np.uint64)
    records["ep"] = -1
    records["move"] = rng.integers(0, 4672, size=count)
    records["wdl"] = rng.integers(0, 3, size=count)
    return records


def dataset_records(data: Path, count: int) -> PackedRecords:
    manifest = read_manifest(data)
    first_shard = manifest.splits[Split.TRAIN].shards[0].file
    return np.array(load_shard(data / first_shard)[:count])


def seconds(step: Callable[[], None], count: int, device: torch.device) -> float:
    """Wall time of ``count`` calls of ``step``, including the GPU work they queue."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(count):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - started


def measure(
    model: str,
    batch_size: int,
    *,
    channels_last: bool,
    compile: bool,
    steps: int,
    warmup: int,
    device: torch.device,
    records: PackedRecords,
) -> Result:
    use_amp = device.type == "cuda"
    gc.collect()
    torch._dynamo.reset()  # every compiled row starts from scratch (no recompile limits)
    if use_amp:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(0)
    config = TrainConfig(model=model, batch_size=batch_size)
    network = ChessNet(MODEL_PRESETS[model]).to(device).train()
    module = prepare_network(network, channels_last=channels_last, compile=compile)
    optimizer = make_optimizer(config, network)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    batch = batch_from_records(records[:batch_size]).to(device)

    def train_step() -> None:
        planes = model_input(unpack_batch_torch(batch), channels_last=channels_last)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            losses = compute_losses(
                module(planes),
                batch.move,
                batch.wdl,
                value_weight=config.value_loss_weight,
                label_smoothing=config.label_smoothing,
            )
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(losses.total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(network.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

    def eval_step() -> None:
        planes = model_input(unpack_batch_torch(batch), channels_last=channels_last)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            network(planes)

    first_step = seconds(train_step, 1, device)
    seconds(train_step, max(warmup - 1, 0), device)
    train_speed = batch_size * steps / seconds(train_step, steps, device)
    memory = torch.cuda.max_memory_allocated(device) / 1e9 if use_amp else math.nan
    eval_speed = math.nan
    if not compile:  # validation always runs the eager network
        network.eval()
        with torch.inference_mode():
            seconds(eval_step, warmup, device)
            eval_speed = batch_size * steps / seconds(eval_step, steps, device)
    return Result(
        model, batch_size, channels_last, compile, train_speed, eval_speed, first_step, memory
    )


def time_loader(data: Path, batch_size: int, batches: int, workers: int) -> float:
    """Positions per second the data loader delivers (CPU side, no model)."""
    dataset = ShardDataset(data, Split.TRAIN, batch_size=batch_size)
    loader = iter(make_loader(dataset, num_workers=workers))
    next(loader)  # worker start-up is not part of the steady state
    started = time.perf_counter()
    delivered = 0
    for _ in range(batches):
        delivered += len(next(loader))
    return delivered / (time.perf_counter() - started)


def on_off(value: bool) -> str:
    return "on" if value else "off"


def speed_text(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:,.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="+", default=["small", "base"], choices=MODEL_PRESETS)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1024, 2048])
    parser.add_argument("--channels-last", nargs="+", default=["off", "on"], choices=SWITCH)
    parser.add_argument("--compile", nargs="+", default=["off", "on"], choices=SWITCH)
    parser.add_argument("--steps", type=int, default=50, help="timed steps per configuration")
    parser.add_argument("--warmup", type=int, default=10, help="untimed steps first")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data", type=Path, help="dataset directory (default: random records)")
    parser.add_argument("--loader-batches", type=int, default=200)
    parser.add_argument("--workers", type=int, default=2, help="data loader workers")
    parser.add_argument("--out", type=Path, help="also write the report to this Markdown file")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True  # as in training (non-deterministic mode)
    largest = max(args.batch_sizes)
    records = dataset_records(args.data, largest) if args.data else random_records(largest)
    train_positions = read_manifest(args.data).splits[Split.TRAIN].positions if args.data else None
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    precision = "fp16 mixed precision" if device.type == "cuda" else "fp32"
    lines = [
        f"# Training throughput ({gpu}, torch {torch.__version__}, {precision})",
        "",
        f"Data: {args.data or 'random records'}. Each row times {args.steps} training steps "
        f"after {args.warmup} warm-up steps; the first step, which includes compilation, is "
        "timed on its own. Eval times forward passes alone, as validation runs them (always "
        "eager, so compiled rows have none).",
        "",
        "| Model | Parameters | Batch | channels_last | compile | Train positions/s "
        "| Eval positions/s | First step | Peak GPU memory | Hours per epoch |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    results: list[Result] = []
    for model in args.models:
        parameters = count_parameters(ChessNet(MODEL_PRESETS[model]))
        for batch_size, channels_last, compile in itertools.product(
            args.batch_sizes, args.channels_last, args.compile
        ):
            row = f"| {model} | {parameters:,} | {batch_size} | {channels_last} | {compile} |"
            try:
                result = measure(
                    model,
                    batch_size,
                    channels_last=SWITCH[channels_last],
                    compile=SWITCH[compile],
                    steps=args.steps,
                    warmup=args.warmup,
                    device=device,
                    records=records,
                )
            except torch.cuda.OutOfMemoryError:
                lines.append(f"{row} out of memory | | | | |")
            except Exception as error:
                if not SWITCH[compile]:
                    raise
                traceback.print_exc()  # a compiler failure must not end the session
                lines.append(f"{row} compile failed: {type(error).__name__} | | | | |")
            else:
                results.append(result)
                hours = train_positions / result.train_speed / 3600 if train_positions else None
                memory = "-" if math.isnan(result.memory) else f"{result.memory:.2f} GB"
                lines.append(
                    f"{row} {speed_text(result.train_speed)} | {speed_text(result.eval_speed)} "
                    f"| {result.first_step:.1f} s | {memory} | "
                    f"{'-' if hours is None else f'{hours:.2f}'} |"
                )
            print(lines[-1], flush=True)

    lines += ["", "Fastest training configuration per model:", ""]
    for model in args.models:
        candidates = [result for result in results if result.model == model]
        if candidates:
            best = max(candidates, key=lambda result: result.train_speed)
            lines.append(
                f"- {model}: batch {best.batch_size}, channels_last {on_off(best.channels_last)}, "
                f"compile {on_off(best.compile)}: {best.train_speed:,.0f} positions/s"
            )
    if args.data:
        lines += ["", "Data loader:", ""]
        for batch_size in args.batch_sizes:
            speed = time_loader(args.data, batch_size, args.loader_batches, args.workers)
            lines.append(f"- {args.workers} workers, batch {batch_size}: {speed:,.0f} positions/s")
    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        args.out.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
