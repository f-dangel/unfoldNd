"""Tests for unfoldNd/__init__.py."""

import time

import pytest

import unfoldNd

NAMES = ["world", "github"]
IDS = NAMES


@pytest.mark.parametrize("name", NAMES, ids=IDS)
def test_hello(name):
    """Test hello function."""
    unfoldNd.hello(name)


@pytest.mark.expensive
@pytest.mark.parametrize("name", NAMES, ids=IDS)
def test_hello_expensive(name):
    """Expensive test of hello function. Will only be run on main and development."""
    time.sleep(1)
    unfoldNd.hello(name)


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
