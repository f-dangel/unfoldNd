"""Tests for ``unfoldNd/unfold_transpose.py.``"""

from test.unfold_transpose_settings import (
    DEVICES,
    DEVICES_ID,
    PROBLEMS_2D,
    PROBLEMS_2D_IDS,
)
from test.utils import _conv_transpose_unfold

import pytest
import torch
from torch.nn import ConvTranspose1d, ConvTranspose2d, ConvTranspose3d

import unfoldNd


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_ID)
@pytest.mark.parametrize("problem", PROBLEMS_2D, ids=PROBLEMS_2D_IDS)
def test_UnfoldTranspose_vs_ConvTransose(problem, device):
    """Compare transpose convolution with matrix-multiplication via unfold."""
    seed = problem["seed"]
    input_fn = problem["input_fn"]
    unfold_transpose_kwargs = problem["unfold_transpose_kwargs"]
    out_channels = problem["out_channels"]
    in_channels = input_fn().shape[1]

    torch.manual_seed(seed)
    inputs = input_fn().to(device)

    conv_transpose_module = {
        1: ConvTranspose1d,
        2: ConvTranspose2d,
        3: ConvTranspose3d,
    }[inputs.dim() - 2]

    conv_transpose_module = conv_transpose_module(
        in_channels, out_channels, **unfold_transpose_kwargs, bias=False
    ).to(device)
    torch_result = conv_transpose_module(inputs)

    unfolded_inputs = unfoldNd.UnfoldTransposeNd(**unfold_transpose_kwargs).to(device)(
        inputs
    )
    result_lib = _conv_transpose_unfold(inputs, unfolded_inputs, conv_transpose_module)

    assert torch.allclose(torch_result, result_lib, atol=5e-7)
