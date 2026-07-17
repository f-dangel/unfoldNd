from torch import allclose, randn
from torch.nn import Module

from unfoldNd import UnfoldNd
from unfoldNd.utils import _get_conv, _get_kernel_size_numel, _tuple


class UnfoldNdONNX(Module):
    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()

        self._kernel_size = kernel_size
        self._dilation = dilation
        self._padding = padding
        self._stride = stride

    def forward(self, input):
        batch_size, in_channels = input.shape[0], input.shape[1]

        # get convolution operation
        batch_size_and_in_channels_dims = 2
        N = input.dim() - batch_size_and_in_channels_dims
        conv = _get_conv(N)

        # prepare one-hot convolution kernel
        kernel_size = _tuple(self._kernel_size, N)
        kernel_size_numel = _get_kernel_size_numel(kernel_size)
        repeat = [in_channels, 1] + [1 for _ in kernel_size]

        weight = (
            torch.eye(kernel_size_numel, device=input.device, dtype=input.dtype)
            .reshape((kernel_size_numel, 1, *kernel_size))
            .repeat(*repeat)
        )

        unfold = conv(
            input,
            weight,
            bias=None,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            groups=in_channels,
        )

        return unfold.reshape(batch_size, in_channels * kernel_size_numel, -1)


b = 2
c = 64
L = 125
step = 3
stride = 2
inputs = randn((b, c, 1, L))
lib_module = UnfoldNd((1, step), dilation=(1, 1), padding=0, stride=stride)
lib_outputs = lib_module(inputs)
a2 = lib_outputs.view(b, c, step, -1).transpose(2, 3)
unfold = inputs.unfold(-1, 3, 2)

assert allclose(lib_outputs, unfold)

torch.onnx.export(
    lib_module,
    inputs,
    "model.onnx",
    input_names=["x"],
    output_names=["y"],
    opset_version=14,
    dynamic_axes={"x": {3: "frame"}},
)
