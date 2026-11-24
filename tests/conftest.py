import os

import pytest
from hypothesis import settings

# At least 200 examples per property test; no per-example deadline (CI machines vary).
settings.register_profile("default", max_examples=200, deadline=None)
# Pull-request CI is derandomized (fixed seed), so a red build always reproduces.
settings.register_profile("ci", parent=settings.get_profile("default"), derandomize=True)
# Nightly: fresh random seeds and ten times the examples (select with --hypothesis-profile).
settings.register_profile("nightly", max_examples=2000, deadline=None, print_blob=True)
settings.load_profile("ci" if os.environ.get("CI") else "default")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--require-gpu",
        action="store_true",
        default=False,
        help="run tests marked 'gpu' even without CUDA, so they fail instead of being skipped",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    gpu_tests = [item for item in items if item.get_closest_marker("gpu")]
    if not gpu_tests or config.getoption("--require-gpu"):
        return

    import torch

    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs a CUDA device (pass --require-gpu to fail instead)")
    for item in gpu_tests:
        item.add_marker(skip)
