"""Tests for ``unfoldNd/unfold.py.``"""

from test.unfold_settings import (
    DEVICES,
    DEVICES_ID,
    PROBLEMS_1D,
    PROBLEMS_1D_IDS,
    PROBLEMS_2D,
    PROBLEMS_2D_IDS,
    PROBLEMS_3D,
    PROBLEMS_3D_IDS,
)
from test.utils import _add_dummy_dim, _conv_unfold

import pytest
import torch

import unfoldNd


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_ID)
@pytest.mark.parametrize("problem", PROBLEMS_2D, ids=PROBLEMS_2D_IDS)
def test_Unfold2d_vs_Unfold(problem, device):
    """Compare with ``torch.nn.Unfold`` for a 4d input."""
    seed = problem["seed"]
    input_fn = problem["input_fn"]
    unfold_kwargs = problem["unfold_kwargs"]

    torch.manual_seed(seed)
    inputs = input_fn().to(device)

    result_torch = torch.nn.Unfold(**unfold_kwargs).to(device)(inputs)
    result_lib = unfoldNd.UnfoldNd(**unfold_kwargs).to(device)(inputs)

    assert torch.allclose(result_lib, result_torch)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_ID)
@pytest.mark.parametrize("problem", PROBLEMS_1D, ids=PROBLEMS_1D_IDS)
def test_Unfold1d_vs_dummy_dim_Unfold(problem, device):
    """Compare with ``torch.nn.Unfold`` for a 3d input (adding a dummy dimension)."""
    seed = problem["seed"]
    input_fn = problem["input_fn"]
    unfold_kwargs = problem["unfold_kwargs"]

    torch.manual_seed(seed)
    inputs = input_fn().to(device)

    unfold_kwargs_dummy_dim, inputs_dummy_dim = _add_dummy_dim(unfold_kwargs, inputs)
    result_torch = torch.nn.Unfold(**unfold_kwargs_dummy_dim).to(device)(
        inputs_dummy_dim
    )

    result_lib = unfoldNd.UnfoldNd(**unfold_kwargs).to(device)(inputs)

    assert torch.allclose(result_lib, result_torch)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_ID)
@pytest.mark.parametrize("problem", PROBLEMS_3D, ids=PROBLEMS_3D_IDS)
def test_Unfold3d_vs_Conv3d(problem, device):
    """
    Use unfolded input in convolution with matrix-view kernel, compare with Conv3d.
    """
    seed = problem["seed"]
    input_fn = problem["input_fn"]
    unfold_kwargs = problem["unfold_kwargs"]
    out_channels = problem["out_channels"]
    in_channels = input_fn().shape[1]

    torch.manual_seed(seed)
    inputs = input_fn().to(device)

    conv3d_module = torch.nn.Conv3d(
        in_channels, out_channels, **unfold_kwargs, bias=False
    ).to(device)
    torch_result = conv3d_module(inputs)

    unfolded_inputs = unfoldNd.UnfoldNd(**unfold_kwargs).to(device)(inputs)
    result_lib = _conv_unfold(inputs, unfolded_inputs, conv3d_module)

    assert torch.allclose(torch_result, result_lib, atol=5e-7)
