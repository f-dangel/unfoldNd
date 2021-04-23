"""Tests for ``unfoldNd/fold.py.`` (fold functionality)."""

from test.fold_settings import DEVICES, DEVICES_ID, PROBLEMS_2D, PROBLEMS_2D_IDS

import pytest
import torch

import unfoldNd


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_ID)
@pytest.mark.parametrize("problem", PROBLEMS_2D, ids=PROBLEMS_2D_IDS)
def test_Fold2d_vs_Fold(problem, device):
    """Compare with ``torch.nn.Fold`` for a 4d input."""
    seed = problem["seed"]
    input_shape = problem["input_shape"]
    fold_kwargs = problem["fold_kwargs"]

    torch.manual_seed(seed)
    inputs = torch.rand(input_shape).to(device)

    result = unfoldNd.FoldNd(**fold_kwargs).to(device)(inputs)
    result_torch = torch.nn.Fold(**fold_kwargs).to(device)(inputs)

    assert torch.allclose(result, result_torch)
