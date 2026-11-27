"""Worker processes: how the CPU is shared out."""

import torch

from chessnet.eval.workers import threads_per_worker


def test_workers_share_torch_s_threads_not_the_logical_cpus() -> None:
    """Regression: one worker got every logical CPU (4 hyper-threads on a 2-core laptop),
    which searched 5-9% slower than torch's default of one thread per physical core."""
    total = torch.get_num_threads()
    assert threads_per_worker(1) == total
    for workers in range(1, 9):
        assert threads_per_worker(workers) >= 1
        assert threads_per_worker(workers) * workers <= max(total, workers)
