"""Tests for unfoldNd/__init__.py."""

from test.settings import (
    DEVICES,
    DEVICES_ID,
    PROBLEMS_1D,
    PROBLEMS_1D_IDS,
    PROBLEMS_2D,
    PROBLEMS_2D_IDS,
    PROBLEMS_3D,
    PROBLEMS_3D_IDS,
    UNSUPPORTED_KERNEL_SIZE,
    UNSUPPORTED_KERNEL_SIZE_IDS,
    UNSUPPORTED_N,
    UNSUPPORTED_N_IDS,
)
from test.utils import _conv_unfold

import pytest
import torch
from torch.nn.modules.utils import _pair

import unfoldNd


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_ID)
@pytest.mark.parametrize("problem", PROBLEMS_2D, ids=PROBLEMS_2D_IDS)
def test_Unfold2d_vs_Unfold(problem, device):
    """Compare with ``torch.nn.Unfold`` for a 4d input."""
    seed = problem["seed"]
    input_shape = problem["input_shape"]
    unfold_kwargs = problem["unfold_kwargs"]

    torch.manual_seed(seed)
    inputs = torch.rand(input_shape).to(device)

    result = unfoldNd.UnfoldNd(**unfold_kwargs).to(device)(inputs)
    result_torch = torch.nn.Unfold(**unfold_kwargs).to(device)(inputs)

    assert torch.allclose(result, result_torch)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_ID)
@pytest.mark.parametrize("problem", PROBLEMS_1D, ids=PROBLEMS_1D_IDS)
def test_Unfold1d_vs_dummy_dim_Unfold(problem, device):
    """Compare with ``torch.nn.Unfold`` for a 3d input (adding a dummy dimension)."""
    seed = problem["seed"]
    input_shape = problem["input_shape"]
    unfold_kwargs = problem["unfold_kwargs"]

    torch.manual_seed(seed)
    inputs = torch.rand(input_shape).to(device)

    result = unfoldNd.UnfoldNd(**unfold_kwargs).to(device)(inputs)

    def _add_dummy_dim(unfold_kwargs, inputs):
        """Add dummy dimension to unfold hyperparameters and input."""
        new_inputs = inputs.unsqueeze(-1)

        new_kwargs = {}

        for key, value in unfold_kwargs.items():
            dummy = (0,) if key == "padding" else (1,)
            new_value = _pair(value)[:-1] + dummy
            new_kwargs[key] = new_value

        return new_kwargs, new_inputs

    unfold_kwargs_dummy_dim, inputs_dummy_dim = _add_dummy_dim(unfold_kwargs, inputs)
    result_torch = torch.nn.Unfold(**unfold_kwargs_dummy_dim).to(device)(
        inputs_dummy_dim
    )

    assert torch.allclose(result, result_torch)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_ID)
@pytest.mark.parametrize("problem", PROBLEMS_3D, ids=PROBLEMS_3D_IDS)
def test_Unfold3d_vs_Conv3d(problem, device):
    """
    Use unfolded input in convolution with matrix-view kernel, compare with Conv3d.
    """
    seed = problem["seed"]
    input_shape = problem["input_shape"]
    unfold_kwargs = problem["unfold_kwargs"]
    out_channels = problem["out_channels"]
    in_channels = input_shape[1]

    torch.manual_seed(seed)
    inputs = torch.rand(input_shape).to(device)

    conv3d_module = torch.nn.Conv3d(
        in_channels, out_channels, **unfold_kwargs, bias=False
    ).to(device)
    torch_result = conv3d_module(inputs)

    unfolded_inputs = unfoldNd.UnfoldNd(**unfold_kwargs).to(device)(inputs)
    result = _conv_unfold(inputs, unfolded_inputs, conv3d_module)

    assert torch.allclose(torch_result, result, atol=5e-7)


@pytest.mark.parametrize("N", UNSUPPORTED_N, ids=UNSUPPORTED_N_IDS)
def test__tuple_raise_dimension_error(N):
    """Only N=1,2,3 are supported."""
    dummy_kernel_size = None

    with pytest.raises(ValueError):
        unfoldNd._tuple(dummy_kernel_size, N)


@pytest.mark.parametrize("N", UNSUPPORTED_N, ids=UNSUPPORTED_N_IDS)
def test__get_conv_raise_dimension_error(N):
    """Only N=1,2,3 are supported."""
    with pytest.raises(ValueError):
        unfoldNd._get_conv(N)


@pytest.mark.parametrize(
    "kernel_size", UNSUPPORTED_KERNEL_SIZE, ids=UNSUPPORTED_KERNEL_SIZE_IDS
)
def test__get_kernel_size_numel_raise_value_error(kernel_size):
    """``kernel_size`` must be an ``N``-tuple."""
    with pytest.raises(ValueError):
        unfoldNd._get_kernel_size_numel(kernel_size)
