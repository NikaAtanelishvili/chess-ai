"""Training throughput benchmark: positions per second for each model preset and batch size.

Times complete training steps (GPU unpacking, forward, backward, optimizer step) with fp16
mixed precision on CUDA, on real records from a dataset (``--data``) or on random records.
With ``--data`` it also measures how fast the data loader delivers batches, and turns the
measured speed into hours per epoch for that dataset.

    python scripts/benchmark_train.py --data /kaggle/input/.../shards --out benchmark.md
"""

from __future__ import annotations

import argparse
import math
import time
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
from chessnet.train.loop import make_optimizer
from chessnet.train.losses import compute_losses


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


def time_training(
    model: str,
    batch_size: int,
    steps: int,
    warmup: int,
    device: torch.device,
    records: PackedRecords,
) -> tuple[float, float]:
    """Training positions per second, and peak GPU memory in GB (NaN on the CPU)."""
    use_amp = device.type == "cuda"
    torch.manual_seed(0)
    network = ChessNet(MODEL_PRESETS[model]).to(device).train()
    optimizer = make_optimizer(TrainConfig(model=model), network)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    batch = batch_from_records(records[:batch_size]).to(device)

    def step() -> None:
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            output = network(unpack_batch_torch(batch))
            loss = compute_losses(
                output, batch.move, batch.wdl, value_weight=0.5, label_smoothing=0.05
            ).total
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

    if use_amp:
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        step()
    if use_amp:
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(steps):
        step()
    if use_amp:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    memory = torch.cuda.max_memory_allocated(device) / 1e9 if use_amp else math.nan
    return batch_size * steps / elapsed, memory


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="+", default=["small", "base"], choices=MODEL_PRESETS)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--steps", type=int, default=50, help="timed steps per configuration")
    parser.add_argument("--warmup", type=int, default=10, help="untimed steps first")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data", type=Path, help="dataset directory (default: random records)")
    parser.add_argument("--loader-batches", type=int, default=200)
    parser.add_argument("--workers", type=int, default=2, help="data loader workers")
    parser.add_argument("--out", type=Path, help="also write the report to this Markdown file")
    args = parser.parse_args()

    device = torch.device(args.device)
    largest = max(args.batch_sizes)
    records = dataset_records(args.data, largest) if args.data else random_records(largest)
    train_positions = read_manifest(args.data).splits[Split.TRAIN].positions if args.data else None
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    lines = [
        f"# Training throughput ({gpu}, torch {torch.__version__}, "
        f"{'fp16 mixed precision' if device.type == 'cuda' else 'fp32'})",
        "",
        f"Data: {args.data or 'random records'}; {args.steps} timed steps after {args.warmup} "
        "warm-up steps.",
        "",
        "| Model | Parameters | Batch | Positions/s | Peak GPU memory | Hours per epoch |",
        "|-------|-----------:|------:|------------:|----------------:|----------------:|",
    ]
    for model in args.models:
        parameters = count_parameters(ChessNet(MODEL_PRESETS[model]))
        for batch_size in args.batch_sizes:
            try:
                speed, memory = time_training(
                    model, batch_size, args.steps, args.warmup, device, records
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                lines.append(f"| {model} | {parameters:,} | {batch_size} | out of memory | | |")
                continue
            hours = f"{train_positions / speed / 3600:.2f}" if train_positions else "-"
            memory_text = "-" if math.isnan(memory) else f"{memory:.2f} GB"
            row = (model, f"{parameters:,}", batch_size, f"{speed:,.0f}", memory_text, hours)
            lines.append("| " + " | ".join(map(str, row)) + " |")
            print(lines[-1], flush=True)
    if args.data:
        for batch_size in args.batch_sizes:
            speed = time_loader(args.data, batch_size, args.loader_batches, args.workers)
            lines.append("")
            label = f"Data loader ({args.workers} workers, batch {batch_size})"
            lines.append(f"{label}: {speed:,.0f} positions/s")
    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        args.out.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
