"""The ``gpu`` marker: skipped without CUDA, but a hard failure under ``--require-gpu``.

The Kaggle training kernel relies on ``pytest -m gpu --require-gpu`` failing loudly when no GPU
is attached, instead of passing with every GPU test skipped.
"""

from pathlib import Path

import pytest
import torch

CONFTEST = (Path(__file__).parents[1] / "conftest.py").read_text(encoding="utf-8")
GPU_TEST = """
import pytest
import torch


@pytest.mark.gpu
def test_needs_cuda():
    torch.zeros(1).to("cuda")
"""


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the behaviour without CUDA")
def test_gpu_tests_skip_without_cuda_unless_required(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(CONFTEST)
    pytester.makeini("[pytest]\nmarkers =\n    gpu: needs a CUDA device\n")
    pytester.makepyfile(GPU_TEST)

    pytester.runpytest_subprocess().assert_outcomes(skipped=1)
    pytester.runpytest_subprocess("--require-gpu").assert_outcomes(failed=1)
    pytester.runpytest_subprocess("-m", "gpu").assert_outcomes(skipped=1)
