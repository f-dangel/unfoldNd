"""unfoldNd library."""

import torch.nn


class UnfoldNd(torch.nn.Module):
    """Extracts sliding local blocks from a batched input tensor. Also known as im2col.

    PyTorch module that accepts 3d, 4d, and 5d tensors. Acts like ``torch.nn.Unfold``
    for a 4d input. Uses one-hot convolution under the hood.

    See docs at https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html.
    """

    def __init__(self, kernel_size, dilation, padding=0, stride=1):
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
    """Extracts sliding local blocks from an batched input tensor. Also known as im2col.

    Pytorch functional that accepts 3d, 4d, and 5d tensors. Acts like
    ``torch.nn.functional.unfold for a 4d input. Uses one-hot convolution under the
    hood.

    See docs at https://pytorch.org/docs/stable/nn.functional.html.
    """
    raise NotImplementedError


def hello(name):
    """Say hello to a name.

    Args:
        name (str): Name to say hello to.
    """
    print(f"Hello, {name}")
