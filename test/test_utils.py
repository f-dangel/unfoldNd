"""Tests for ``unfoldNd/utils.py.``"""

from test.utils_settings import (
    UNSUPPORTED_KERNEL_SIZE,
    UNSUPPORTED_KERNEL_SIZE_IDS,
    UNSUPPORTED_N,
    UNSUPPORTED_N_IDS,
)

import pytest

from unfoldNd import utils


@pytest.mark.parametrize("N", UNSUPPORTED_N, ids=UNSUPPORTED_N_IDS)
def test__tuple_raise_dimension_error(N):
    """Only N=1,2,3 are supported."""
    dummy_kernel_size = None

    with pytest.raises(ValueError):
        utils._tuple(dummy_kernel_size, N)


@pytest.mark.parametrize("N", UNSUPPORTED_N, ids=UNSUPPORTED_N_IDS)
def test__get_conv_raise_dimension_error(N):
    """Only N=1,2,3 are supported."""
    with pytest.raises(ValueError):
        utils._get_conv(N)


@pytest.mark.parametrize("N", UNSUPPORTED_N, ids=UNSUPPORTED_N_IDS)
def test__get_conv_transpose_raise_dimension_error(N):
    """Only N=1,2,3 are supported."""
    with pytest.raises(ValueError):
        utils._get_conv_transpose(N)


@pytest.mark.parametrize(
    "kernel_size", UNSUPPORTED_KERNEL_SIZE, ids=UNSUPPORTED_KERNEL_SIZE_IDS
)
def test__get_kernel_size_numel_raise_value_error(kernel_size):
    """``kernel_size`` must be an ``N``-tuple."""
    with pytest.raises(ValueError):
        utils._get_kernel_size_numel(kernel_size)
