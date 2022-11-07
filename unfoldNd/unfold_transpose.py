"""Generalization of unfold operation for transpose convolutions."""

import torch

from unfoldNd.utils import (
    _get_conv,
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

        Note:
            This may break if the PyTorch code changes.

        Link:
            https://github.com/pytorch/pytorch/blob/febff45900e57d3e05ee72c1ecfe7d4fcbc582d9/torch/nn/modules/conv.py#L606-L644.
        """
        # get convolution operation
        batch_size_and_in_channels_dims = 2
        N = input.dim() - batch_size_and_in_channels_dims

        # TODO Required for _output_padding
        kernel_size = _tuple(kernel_size, N)
        stride = _tuple(stride, N)
        padding = _tuple(padding, N)
        dilation = _tuple(dilation, N)

        if output_size is None:
            return _tuple(self._output_padding, N)

        self_dummy = None

        return torch.nn.modules.conv._ConvTransposeNd._output_padding(
            self_dummy, input, output_size, stride, padding, kernel_size, dilation
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
    # TODO Update docstring
    # TODO Maybe recycle one-hot weight from conv
    """Create one-hot transpose convolution kernel. ``kernel_size`` must be an ``N``-tuple.

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

    # TODO Write more compactly
    weight = torch.zeros(1, kernel_size_numel, *kernel_size)

    for i in range(kernel_size_numel):
        extraction = torch.zeros(kernel_size_numel)
        extraction[i] = 1.0
        weight[0, i] = extraction.reshape(*kernel_size)

    weight = weight.repeat(*repeat)
    return weight.to(dtype).to(device)

    # return (
    #     torch.eye(kernel_size_numel, device=device, dtype=dtype)
    #     .reshape((kernel_size_numel, 1, *kernel_size))
    #     .repeat(*repeat)
    # )
