"""Tests for unfoldNd/__init__.py."""

import pytest

import unfoldNd

input = None
kernel_size = None
dilation = 1
padding = 0
stride = 1


def test_unfoldNd_not_implemented():
    """``unfoldNd.unfoldNd`` is currently not implemented.

    Raises:
        NotImplementedError : always.
    """
    with pytest.raises(NotImplementedError):
        unfoldNd.unfoldNd(
            input, kernel_size, dilation=dilation, padding=padding, stride=stride
        )


def test_UnfoldNd_forward_not_implemented():
    """``unfoldNd.UnfoldNd.forward`` is currently not implemented.

    Raises:
        NotImplementedError : always.
    """
    module = unfoldNd.UnfoldNd(
        kernel_size, dilation=dilation, padding=padding, stride=stride
    )

    with pytest.raises(NotImplementedError):
        module(input)


UNSUPPORTED_N = [4, -1]


@pytest.mark.parametrize("N", UNSUPPORTED_N)
def test__tuple_raise_dimension_error(N):
    """Only N=1,2,3 are supported."""
    dummy_kernel_size = None

    with pytest.raises(ValueError):
        unfoldNd._tuple(dummy_kernel_size, N)


@pytest.mark.parametrize("N", UNSUPPORTED_N)
def test__get_conv_raise_dimension_error(N):
    """Only N=1,2,3 are supported."""
    with pytest.raises(ValueError):
        unfoldNd._get_conv(N)


UNSUPPORTED_KERNEL_SIZE = [[1, 2], 1]


@pytest.mark.parametrize("kernel_size", UNSUPPORTED_KERNEL_SIZE)
def test__get_kenel_size_numel_raise_value_error(kernel_size):
    """``kernel_size`` must be an ``N``-tuple."""
    with pytest.raises(ValueError):
        unfoldNd._get_kernel_size_numel(kernel_size)
