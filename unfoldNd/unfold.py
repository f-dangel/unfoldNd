"""Generalization of unfold operation."""

import torch

from unfoldNd.utils import _get_conv, _get_kernel_size_numel, _tuple


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
    ``torch.nn.functional.unfold`` for a 4d input. Uses one-hot convolution under the
    hood.

    See docs at https://pytorch.org/docs/stable/nn.functional.html#unfold.
    """
    batch_size, in_channels = input.shape[0], input.shape[1]

    # get convolution operation
    batch_size_and_in_channels_dims = 2
    N = input.dim() - batch_size_and_in_channels_dims
    conv = _get_conv(N)

    # prepare one-hot convolution kernel
    kernel_size = _tuple(kernel_size, N)
    kernel_size_numel = _get_kernel_size_numel(kernel_size)
    weight = _make_weight(in_channels, kernel_size, input.device, input.dtype)

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


def _make_weight(in_channels, kernel_size, device, dtype):
    """Create one-hot convolution kernel. ``kernel_size`` must be an ``N``-tuple.

    Details:
        Let's assume 2d convolution. We are given an input of shape `[N,
        C_in, H, W]` and a kernel of shape `[*, *, K_H, K_W]`. We then want to
        produce an output with shape `[N, C_in * K_H * K_W, L]` with `L` the
        number of patches. We can run convolution with `groups=C_in`. This will
        treat each input channel independently with the same kernel `t` of shape
        `[K_H * K_W, 1, K_H, K_W]` that satisfies `t[h * w, 0, h, w] = δ_{h, w}`.
        We can run convolution with `groups=C_in` to achieve this independent
        treatment, but for that we must duplicate it `C_in` times along the leading
        dimension, because the kernel's output dimension must match that of the output
        for convolution in group mode (see its documentation).

        This yields a kernel `T` that satisfies `T[c * h * w, 0, h, w] = δ_{h,w}`.

        Such a kernel is formed by creating a `K_H * K_W` identity matrix,
        reshaping it into `[K_H * K_W, 1, K_H, K_W]` (`t`), and repeating it `C_in`
        times along the leading dimension (`T`).

    Returns:
        torch.Tensor : A tensor of shape ``[C_in * ∏ᵢ Kᵢ, 1, K]`` where
            ``K = (K₁, K₂, ..., Kₙ)`` is the kernel size. Filter groups are
            one-hot such that they effectively extract one element of the patch
            the kernel currently overlaps with.
    """
    kernel_size_numel = _get_kernel_size_numel(kernel_size)
    repeat = [in_channels, 1] + [1 for _ in kernel_size]

    return (
        torch.eye(kernel_size_numel, device=device, dtype=dtype)
        .reshape((kernel_size_numel, 1, *kernel_size))
        .repeat(*repeat)
    )
