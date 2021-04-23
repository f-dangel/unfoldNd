"""Generalization of fold operation."""

import torch

from unfoldNd.unfold import unfoldNd
from unfoldNd.utils import _get_kernel_size_numel, _get_numel_from_shape, _tuple


class FoldNd(torch.nn.Module):
    """Combines an array of sliding local blocks into a large containing tensor.

    Also known as col2im.

    PyTorch module that accepts 3d, 4d, and 5d tensors. Acts like ``torch.nn.Fold``
    for a 4d input. Unfolds an index tensor to scatter sliding blocks to the result.

    See docs at https://pytorch.org/docs/stable/generated/torch.nn.Fold.html.
    """

    def __init__(self, output_size, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()

        self._output_size = output_size
        self._kernel_size = kernel_size
        self._dilation = dilation
        self._padding = padding
        self._stride = stride

    def forward(self, input):
        return foldNd(
            input,
            self._output_size,
            self._kernel_size,
            dilation=self._dilation,
            padding=self._padding,
            stride=self._stride,
        )


def foldNd(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    """Combines an array of sliding local blocks into a large containing tensor.

    Also known as col2im.

    Pytorch functional that accepts 3d, 4d, and 5d tensors. Acts like
    ``torch.nn.functional.fold for a 4d input. Unfolds an index tensor to scatter
    sliding blocks to the result.

    See docs at
    https://pytorch.org/docs/master/generated/torch.nn.functional.fold.html.

    Raises:
        ValueError: If ``output_size`` is not specified as ``tuple``. Otherwise
            the fold dimension ``N`` cannot be inferred.
    """
    device = input.device

    if isinstance(output_size, tuple):
        N = len(output_size)
        output_size_numel = _get_numel_from_shape(output_size)
    else:
        raise ValueError(f"Expect 'output_size' to be tuple. Got {type(output_size)}")

    kernel_size = _tuple(kernel_size, N)
    kernel_size_numel = _get_kernel_size_numel(kernel_size)

    batch_size = input.shape[0]
    in_channels_kernel_size_numel = input.shape[1]
    in_channels = in_channels_kernel_size_numel // kernel_size_numel

    # NOTE unfolding acts on float tensors. Special attention has to be paid if the
    # arange exceeds the smallest integer that cannot be represented as float32
    # (see https://stackoverflow.com/q/27207149 for details)
    # TODO Use float64 if float32 cannot represent output_size_numel
    idx = torch.arange(output_size_numel, dtype=torch.float32, device=device).reshape(
        1, 1, *output_size
    )
    idx_unfold = unfoldNd(
        idx, kernel_size, dilation=dilation, padding=padding, stride=stride
    )

    input = input.reshape(batch_size, in_channels, -1)
    idx_unfold = idx_unfold.reshape(1, 1, -1).long().expand(batch_size, in_channels, -1)

    output = torch.zeros(
        batch_size, in_channels, output_size_numel, device=device, dtype=input.dtype
    )
    output.scatter_add_(2, idx_unfold, input)

    return output.reshape(batch_size, in_channels, *output_size)
