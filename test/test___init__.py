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
