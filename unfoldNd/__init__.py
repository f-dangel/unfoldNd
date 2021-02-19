"""unfoldNd library."""


import numpy
import torch
from torch.nn.functional import conv1d, conv2d, conv3d
from torch.nn.modules.utils import _pair, _single, _triple

__all__ = ["UnfoldNd", "unfoldNd"]


class UnfoldNd(torch.nn.Module):
    """Extracts sliding local blocks from a batched input tensor. Also known as im2col.

    PyTorch module that accepts 3d, 4d, and 5d tensors. Acts like ``torch.nn.Unfold``
    for a 4d input. Uses one-hot convolution under the hood.

    See docs at https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html.
    """

    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()

        self._kernel_size = kernel_size
        self._dilation = dilation
        self._padding = padding
        self._stride = stride

    def forward(self, input):
        return unfoldNd(
            input,
            self._kernel_size,
            dilation=self._dilation,
            padding=self._padding,
            stride=self._stride,
        )


def unfoldNd(input, kernel_size, dilation=1, padding=0, stride=1):
    """Extracts sliding local blocks from a batched input tensor. Also known as im2col.

    Pytorch functional that accepts 3d, 4d, and 5d tensors. Acts like
    ``torch.nn.functional.unfold for a 4d input. Uses one-hot convolution under the
    hood.

    See docs at https://pytorch.org/docs/stable/nn.functional.html.
    """
    batch_size, in_channels = input.shape[0], input.shape[1]

    # get convolution operation
    batch_size_and_in_channels_dims = 2
    N = input.dim() - batch_size_and_in_channels_dims
    conv = _get_conv(N)

    # prepare one-hot convolution kernel
    kernel_size = _tuple(kernel_size, N)
    kernel_size_numel = _get_kernel_size_numel(kernel_size)
    weight = _make_weight(in_channels, kernel_size, device=input.device)

    unfold = conv(
        input,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=in_channels,
    )

    return unfold.reshape(batch_size, in_channels * kernel_size_numel, -1)


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


def _get_conv(N):
    """Return convolution operation used to perform unfolding."""
    if N == 1:
        return conv1d
    elif N == 2:
        return conv2d
    if N == 3:
        return conv3d
    else:
        _raise_dimension_error(N)


def _raise_dimension_error(N):
    """Notify user that inferred input dimension is not supported."""
    raise ValueError(f"Only 1,2,3-dimensional unfold is supported. Got N={N}.")


def _make_weight(in_channels, kernel_size, device):
    """Create one-hot convolution kernel. ``kernel_size`` must be an ``N``-tuple.

    Details:
        Let ``T`` denote the one-hot weight, then
        ``T[c * i, 0, j] = δᵢⱼ ∀ c = 1, ... C_in``
        (``j`` is a group index of the ``Kᵢ``).

        This can be done by building diagonals ``D[i, j] = δᵢⱼ``, reshaping
        them into ``[∏ᵢ Kᵢ, 1, K]``, and repeat them ``C_in`` times along the
        leading dimension.

    Returns:
        torch.Tensor : A tensor of shape ``[ C_in * ∏ᵢ Kᵢ, 1, K]`` where
            ``K = (K₁, K₂, ..., Kₙ)`` is the kernel size. Filter groups are
            one-hot such that they effectively extract one element of the patch
            the kernel currently overlaps with.


    """
    kernel_size_numel = _get_kernel_size_numel(kernel_size)
    repeat = [in_channels, 1] + [1 for _ in kernel_size]

    return (
        torch.eye(kernel_size_numel, device=device)
        .reshape((kernel_size_numel, 1, *kernel_size))
        .repeat(*repeat)
    )


def _get_kernel_size_numel(kernel_size):
    """Determine number of pixels/voxels. ``kernel_size`` must be an ``N``-tuple. """
    if not isinstance(kernel_size, tuple):
        raise ValueError(f"kernel_size must be a tuple. Got {kernel_size}.")

    return numpy.prod(kernel_size)
