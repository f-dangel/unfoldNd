"""Shared utility functions."""

from packaging.version import Version
from importlib.metadata import version
from typing import Callable

import numpy
from torch.nn.functional import (
    conv1d,
    conv2d,
    conv3d,
    conv_transpose1d,
    conv_transpose2d,
    conv_transpose3d,
)
from torch.nn.modules.utils import _pair, _single, _triple

TORCH_VERSION = Version(version("torch"))
TORCH_VERSION_AT_LEAST_1_12_0 = TORCH_VERSION >= Version("1.12.0")


def _get_kernel_size_numel(kernel_size):
    """Determine number of pixels/voxels. ``kernel_size`` must be an ``N``-tuple."""
    if not isinstance(kernel_size, tuple):
        raise ValueError(f"kernel_size must be a tuple. Got {kernel_size}.")

    return _get_numel_from_shape(kernel_size)


def _get_numel_from_shape(shape_tuple):
    """Compute number of elements from shape."""
    return int(numpy.prod(shape_tuple))


def _tuple(kernel_size, N):
    """Turn ``kernel_size`` argument of ``N``d convolution into an ``N``-tuple."""
    if N == 1:
        return _single(kernel_size)
    elif N == 2:
        return _pair(kernel_size)
    elif N == 3:
        return _triple(kernel_size)
    else:
        _raise_dimension_error(N)


def _get_conv(N: int) -> Callable:
    """Return convolution operation used to perform unfolding."""
    if N == 1:
        return conv1d
    elif N == 2:
        return conv2d
    elif N == 3:
        return conv3d
    else:
        _raise_dimension_error(N)


def _get_conv_transpose(N: int) -> Callable:
    """Return transpose convolution operation used to perform unfolding."""
    if N == 1:
        return conv_transpose1d
    elif N == 2:
        return conv_transpose2d
    elif N == 3:
        return conv_transpose3d
    else:
        _raise_dimension_error(N)


def _raise_dimension_error(N):
    """Notify user that inferred input dimension is not supported."""
    raise ValueError(f"Only 1,2,3-dimensional unfold is supported. Got N={N}.")
