"""Tests for ``unfoldNd/fold.py.`` (fold functionality)."""

from test.fold_settings import (
    DEVICES,
    DEVICES_ID,
    PRECISION_PROBLEMS_2D,
    PRECISION_PROBLEMS_2D_IDS,
    PROBLEMS_2D,
    PROBLEMS_2D_IDS,
    PROBLEMS_INVERSE,
    PROBLEMS_INVERSE_IDS,
    UNSUPPORTED_ARGS_PROBLEMS,
    UNSUPPORTED_ARGS_PROBLEMS_IDS,
)
from test.unfold_settings import PROBLEMS_1D as UNFOLD_PROBLEMS_1D
from test.unfold_settings import PROBLEMS_1D_IDS as UNFOLD_PROBLEMS_1D_IDS
from test.unfold_settings import PROBLEMS_2D as UNFOLD_PROBLEMS_2D
from test.unfold_settings import PROBLEMS_2D_IDS as UNFOLD_PROBLEMS_2D_IDS
from test.unfold_settings import PROBLEMS_3D as UNFOLD_PROBLEMS_3D
from test.unfold_settings import PROBLEMS_3D_IDS as UNFOLD_PROBLEMS_3D_IDS
from test.utils import _add_dummy_dim

import pytest
import torch

import unfoldNd


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_ID)
@pytest.mark.parametrize(
    "problem", UNSUPPORTED_ARGS_PROBLEMS, ids=UNSUPPORTED_ARGS_PROBLEMS_IDS
)
def test_FoldNd_unsupported_args(problem, device):
    """Check unsupported arguments of ``FoldNd``."""
    seed = problem["seed"]
    input_fn = problem["input_fn"]
    fold_kwargs = problem["fold_kwargs"]

    torch.manual_seed(seed)
    inputs = input_fn().to(device)

    with pytest.raises(ValueError):
        _ = unfoldNd.FoldNd(**fold_kwargs).to(device)(inputs)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_ID)
@pytest.mark.parametrize("problem", PROBLEMS_2D, ids=PROBLEMS_2D_IDS)
def test_Fold2d_vs_Fold(problem, device):
    """Compare with ``torch.nn.Fold`` for a 4d input."""
    seed = problem["seed"]
    input_fn = problem["input_fn"]
    fold_kwargs = problem["fold_kwargs"]

    torch.manual_seed(seed)
    inputs = input_fn().to(device)

    result_torch = torch.nn.Fold(**fold_kwargs).to(device)(inputs)
    result_lib = unfoldNd.FoldNd(**fold_kwargs).to(device)(inputs)

    assert torch.allclose(result_lib, result_torch)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_ID)
@pytest.mark.parametrize(
    "problem", PRECISION_PROBLEMS_2D, ids=PRECISION_PROBLEMS_2D_IDS
)
def test_Fold2d_vs_Fold_precision(problem, device):
    """Catch expected shortcomings of ``FoldNd`` caused by unfolding float indices."""
    seed = problem["seed"]
    input_fn = problem["input_fn"]
    fold_kwargs = problem["fold_kwargs"]

    torch.manual_seed(seed)
    inputs = input_fn().to(device)

    _ = torch.nn.Fold(**fold_kwargs).to(device)(inputs)

    with pytest.raises(RuntimeError):
        _ = unfoldNd.FoldNd(**fold_kwargs).to(device)(inputs)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_ID)
@pytest.mark.parametrize("problem", UNFOLD_PROBLEMS_2D, ids=UNFOLD_PROBLEMS_2D_IDS)
def test_Fold2d_vs_Fold_after_Unfold(problem, device):
    """Compare with ``torch.nn.Fold`` for a 4d input.

    Generate settings from unfold tests.
    """
    seed = problem["seed"]
    input_fn = problem["input_fn"]
    unfold_kwargs = problem["unfold_kwargs"]

    torch.manual_seed(seed)
    unfold_input = input_fn().to(device)
    inputs = torch.nn.functional.unfold(unfold_input, **unfold_kwargs)

    fold_kwargs = problem["unfold_kwargs"]
    output_size = unfold_input.shape[2:]

    result_torch = torch.nn.Fold(output_size, **fold_kwargs).to(device)(inputs)
    result_lib = unfoldNd.FoldNd(output_size, **fold_kwargs).to(device)(inputs)

    assert torch.allclose(result_lib, result_torch)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_ID)
@pytest.mark.parametrize("problem", UNFOLD_PROBLEMS_1D, ids=UNFOLD_PROBLEMS_1D_IDS)
def test_Fold1d_vs_Fold_after_dummy_dim_Unfold(problem, device):
    """Compare with ``torch.nn.Fold`` for a 3d input.

    Generate settings from unfold tests and by adding a dummy dimension to achieve
    compatibility with ``torch.nn.Unfold``.
    """
    seed = problem["seed"]
    input_fn = problem["input_fn"]
    unfold_kwargs = problem["unfold_kwargs"]

    torch.manual_seed(seed)
    unfold_inputs = input_fn().to(device)

    unfold_kwargs_dummy_dim, inputs_dummy_dim = _add_dummy_dim(
        unfold_kwargs, unfold_inputs
    )
    inputs = torch.nn.Unfold(**unfold_kwargs_dummy_dim).to(device)(inputs_dummy_dim)

    output_size_dummy_dim = tuple(inputs_dummy_dim.shape[2:])

    result_torch = (
        torch.nn.Fold(output_size_dummy_dim, **unfold_kwargs_dummy_dim)
        .to(device)(inputs)
        .squeeze(-1)
    )

    fold_kwargs = problem["unfold_kwargs"]
    output_size = unfold_inputs.shape[2:]
    result_lib = unfoldNd.FoldNd(output_size, **fold_kwargs).to(device)(inputs)

    assert torch.allclose(result_lib, result_torch)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_ID)
@pytest.mark.parametrize("problem", PROBLEMS_INVERSE, ids=PROBLEMS_INVERSE_IDS)
def test_Fold_inverse_of_Unfold(problem, device):
    """Compare that folding is the inverse of unfolding on 3d/4d/5d inputs.

    This relation only holds if every pixel/voxel is used exactly once, i.e.
    patches don't overlap and cover the entire image/volume.
    """
    seed = problem["seed"]
    input_fn = problem["input_fn"]
    unfold_kwargs = problem["unfold_kwargs"]

    torch.manual_seed(seed)
    inputs = input_fn().to(device)
    unfolded = unfoldNd.unfoldNd(inputs, **unfold_kwargs)

    fold_kwargs = problem["unfold_kwargs"]
    output_size = inputs.shape[2:]

    folded = unfoldNd.FoldNd(output_size, **fold_kwargs).to(device)(unfolded)

    assert torch.allclose(inputs, folded)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_ID)
@pytest.mark.parametrize(
    "problem",
    UNFOLD_PROBLEMS_1D + UNFOLD_PROBLEMS_2D + UNFOLD_PROBLEMS_3D,
    ids=UNFOLD_PROBLEMS_1D_IDS + UNFOLD_PROBLEMS_2D_IDS + UNFOLD_PROBLEMS_3D_IDS,
)
def test_FoldNd_divisor(problem, device):
    """Test divisor tensor from ``fold-unfold`` composition.

    According to https://pytorch.org/docs/stable/generated/torch.nn.Fold.html the
    divisor between an input tensor and the result of an unfold-fold composition
    is satisfies ``fold(unfold(input)) == divisor * input`` with
    ``input_ones = torch.ones(input.shape, dtype=input.dtype)`` and
    ``divisor = fold(unfold(input_ones))``
    """
    seed = problem["seed"]
    input_fn = problem["input_fn"]
    unfold_kwargs = problem["unfold_kwargs"]

    torch.manual_seed(seed)
    inputs = input_fn().to(device)
    inputs_ones = torch.ones(inputs.shape, dtype=inputs.dtype).to(device)

    unfold_module = unfoldNd.UnfoldNd(**unfold_kwargs).to(device)

    fold_kwargs = problem["unfold_kwargs"]
    output_size = inputs.shape[2:]
    fold_module = unfoldNd.FoldNd(output_size, **fold_kwargs).to(device)

    divisor = fold_module(unfold_module(inputs_ones))
    outputs = fold_module(unfold_module(inputs))

    assert torch.allclose(outputs, divisor * inputs)
