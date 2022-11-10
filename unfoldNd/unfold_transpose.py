"""Generalization of unfold operation for transpose convolutions."""

import torch

from unfoldNd.utils import (
    TORCH_VERSION_AT_LEAST_1_12_0,
    _get_conv_transpose,
    _get_kernel_size_numel,
    _tuple,
)


class UnfoldTransposeNd(torch.nn.Module):
    """Extracts sliding local blocks from a batched input tensor. Also known as im2col.

    PyTorch module that accepts 3d, 4d, and 5d tensors. Acts similar to
    ``torch.nn.Unfold``, but assumes transpose convolution to compute sliding blocks.
    Uses one-hot transpose convolution under the hood.

    See docs at
    https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html.
    """

    def __init__(
        self,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
    ):
        super().__init__()

        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._output_padding = output_padding
        self._dilation = dilation

    def _get_output_padding(
        self, input, output_size, stride, padding, kernel_size, dilation
    ):
        """Determine ``output_padding`` from hyperparameters and ``output_size``.

        This follows the hacky way described in
        https://discuss.pytorch.org/t/the-output-size-of-convtranspose2d-differs-from-the-expected-output-size/1876/11 # noqa: B950

        Note:
            This may break if the PyTorch code changes. Please submit an issue if
            you encounter problems.

        Link:
            https://github.com/pytorch/pytorch/blob/febff45900e57d3e05ee72c1ecfe7d4fcbc582d9/torch/nn/modules/conv.py#L606-L644. # noqa: B950
        """
        # get convolution operation
        batch_size_and_in_channels_dims = 2
        N = input.dim() - batch_size_and_in_channels_dims

        # get _output_padding
        kernel_size = _tuple(kernel_size, N)
        stride = _tuple(stride, N)
        padding = _tuple(padding, N)
        dilation = _tuple(dilation, N)

        if output_size is None:
            return _tuple(self._output_padding, N)

        self_dummy = None

        # the signature of _output padding changed between torch==1.11.1 and 1.12.0
        if TORCH_VERSION_AT_LEAST_1_12_0:
            return torch.nn.modules.conv._ConvTransposeNd._output_padding(
                self_dummy,
                input,
                output_size,
                stride,
                padding,
                kernel_size,
                N,
                dilation=dilation,
            )

        return torch.nn.modules.conv._ConvTransposeNd._output_padding(
            self_dummy,
            input,
            output_size,
            stride,
            padding,
            kernel_size,
            dilation=dilation,
        )

    def forward(self, input, output_size=None):
        output_padding = self._get_output_padding(
            input,
            output_size,
            self._stride,
            self._padding,
            self._kernel_size,
            self._dilation,
        )

        return unfold_transposeNd(
            input,
            self._kernel_size,
            stride=self._stride,
            padding=self._padding,
            output_padding=output_padding,
            dilation=self._dilation,
        )


def unfold_transposeNd(
    input, kernel_size, stride=1, padding=0, output_padding=0, dilation=1
):
    """Extracts sliding local blocks from a batched input tensor. Also known as im2col.

    Pytorch functional that accepts 3d, 4d, and 5d tensors. Acts similar to
    ``torch.nn.functional.unfold``, but assumes transpose convolution to compute
    sliding blocks. Uses one-hot transpose convolution under the hood.

    See docs at https://pytorch.org/docs/stable/nn.functional.html#unfold.
    """
    batch_size, in_channels = input.shape[0], input.shape[1]

    # get transpose convolution operation
    batch_size_and_in_channels_dims = 2
    N = input.dim() - batch_size_and_in_channels_dims
    conv_transpose = _get_conv_transpose(N)

    # prepare one-hot transpose convolution kernel
    kernel_size = _tuple(kernel_size, N)
    kernel_size_numel = _get_kernel_size_numel(kernel_size)
    weight = _make_weight(in_channels, kernel_size, input.device, input.dtype)

    unfold = conv_transpose(
        input,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=in_channels,
    )

    return unfold.reshape(batch_size, in_channels * kernel_size_numel, -1)


def _make_weight(in_channels, kernel_size, device, dtype):
    """Create one-hot transpose convolution kernel.

    ``kernel_size`` must be an ``N``-tuple.

    This is basically the same construction mechanism as for unfolding for convolutions.
    For kernels of transpose convolution however, the dimensions of in_channels and
    out_channels are swapped.

    Details: Let's assume 2d convolution. We are given an input of shape `[N,
        C_in, H, W]` and a kernel of hape [*, *, K_H, K_W]`. We then want to
        produce an output with shape `[N, C_in * K_H * K_W, L]` with `L` the
        patch size. We want to treat each input channel independently with the
        same kernel `t` of shape `[1, K_H * K_W, K_H, K_W]` that satisfies
        `t[0, h * w, h, w] = δ_{h,w}`. We can run transpose convolution with
        `groups=C_in` to achieve this independent treatment. Because the
        kernel's input channels must match that of the input for transpose
        convolution (see its documentation), we need to replicate this kernel
        for each channel (`C_in` times) along the leading (input channel) axes.

        This yields a kernel `T` that satisfies `T[c, h * w, h, w] = δ_{h, w}`.

        Such a kernel is formed by creating a `K_H * K_W` identity matrix,
        reshaping it into `[1, K_H * K_W, K_H, K_W]`, and repeating it `C_in`
        times along the leading dimension.

    Returns:
        torch.Tensor : A tensor of shape ``[C_in, ∏ᵢ Kᵢ, K]`` where
            ``K = (K₁, K₂, ..., Kₙ)`` is the kernel size. Filter groups are
            one-hot such that they effectively extract one element of the patch
            the kernel currently overlaps with.
    """
    kernel_size_numel = _get_kernel_size_numel(kernel_size)
    repeat = [in_channels, 1] + [1 for _ in kernel_size]

    return (
        torch.eye(kernel_size_numel, device=device, dtype=dtype)
        .reshape((1, kernel_size_numel, *kernel_size))
        .repeat(*repeat)
    )
