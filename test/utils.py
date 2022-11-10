"""Utility functions for testing ``unfoldNd``."""

import torch
from torch.nn.modules.utils import _pair


def make_id(problem):
    """Convert problem description in to human-readable id."""
    key_value_strs = []

    for key, value in problem.items():
        if key == "input_fn":
            key_value_strs.append(f"input_shape={value().shape}")
        else:
            key_value_strs.append(f"{key}={value}")

    return ",".join(key_value_strs).replace(" ", "")


def get_available_devices():
    """Return CPU and, if present, GPU device.

    Returns:
        [torch.device]: Available devices for `torch`.
    """
    devices = [torch.device("cpu")]

    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    return devices


def _conv_unfold(inputs, unfolded_input, conv_module):
    """Perform convolution with unfolded input via matrix multiplication.

    Copied and modified from:
        https://github.com/f-dangel/backpack/blob/development/test/utils/test_conv.py#L23-L51 # noqa: B950
    """
    assert conv_module.bias is None

    def get_output_shape(inputs, module):
        return module(inputs).shape

    N, C_in = inputs.shape[0], inputs.shape[1]

    output_shape = get_output_shape(inputs, conv_module)
    C_out = output_shape[1]
    spatial_out_size = output_shape[2:]
    spatial_out_numel = spatial_out_size.numel()

    kernel_size = conv_module.kernel_size
    kernel_size_numel = int(torch.prod(torch.Tensor(kernel_size)))

    G = conv_module.groups

    weight_matrix = conv_module.weight.data.reshape(
        G, C_out // G, C_in // G, kernel_size_numel
    )
    unfolded_input = unfolded_input.reshape(
        N, G, C_in // G, kernel_size_numel, spatial_out_numel
    )

    result = torch.einsum("gocx,ngcxh->ngoh", weight_matrix, unfolded_input)

    return result.reshape(N, C_out, *spatial_out_size)


def _conv_transpose_unfold(
    inputs, unfolded_input, conv_transpose_module, output_size=None
):
    """Perform transpose convolution via matrix multiplication.

    Copied and modified from:
        https://github.com/f-dangel/backpack/blob/development/test/utils/test_conv_transpose.py#L17-L43 # noqa: B950
    """
    assert conv_transpose_module.bias is None

    def get_output_shape(input, module, output_size):
        return module(input, output_size=output_size).shape

    N, C_in = inputs.shape[0], inputs.shape[1]

    output_shape = get_output_shape(inputs, conv_transpose_module, output_size)
    C_out = output_shape[1]
    spatial_out_size = output_shape[2:]
    spatial_out_numel = spatial_out_size.numel()
    kernel_size_numel = conv_transpose_module.weight.shape[2:].numel()

    G = conv_transpose_module.groups

    weight_matrix = conv_transpose_module.weight.data.reshape(
        C_in // G, G, C_out // G, kernel_size_numel
    )
    unfolded_input = unfolded_input.reshape(
        N, C_in // G, G, kernel_size_numel, spatial_out_numel
    )

    result = torch.einsum("cgox,ncgxh->ngoh", weight_matrix, unfolded_input)

    return result.reshape(N, C_out, *spatial_out_size)


def _add_dummy_dim(unfold_kwargs, inputs):
    """Add dummy dimension to unfold hyperparameters and input."""
    new_inputs = inputs.unsqueeze(-1)

    new_kwargs = {}

    for key, value in unfold_kwargs.items():
        dummy = (0,) if key == "padding" else (1,)
        new_value = _pair(value)[:-1] + dummy
        new_kwargs[key] = new_value

    return new_kwargs, new_inputs
