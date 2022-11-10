"""How to use ``unfoldTransposeNd``. Relation to ``torch.nn.ConvTransposeNd``."""

# imports, make this example deterministic
import torch

import unfoldNd

torch.manual_seed(0)

# random batched RGB 10x10 image-shaped input tensor of batch size 64
in_channels = 3
inputs = torch.randn((64, in_channels, 10, 10))

# Let's create a transpose convolution module and feed the input

# module hyperparameters
out_channels = 4
output_padding = 0
kernel_size = 2
dilation = 1
padding = 0
stride = 2

# forward pass hyperparameters
output_size = (21, 21)

# compute the result of transpose convolution
conv_transpose_module = torch.nn.ConvTranspose2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=stride,
    padding=padding,
    output_padding=output_padding,
    bias=False,
)
conv_t = conv_transpose_module(inputs, output_size=output_size)

# Now, let's compute the output through matrix-multiplication of the kernel and
# the unfolded input

# compute the unfolded input
unfold_transpose_module = unfoldNd.UnfoldTransposeNd(
    kernel_size,
    stride=stride,
    padding=padding,
    output_padding=output_padding,
    dilation=dilation,
)
inputs_unfolded = unfold_transpose_module(inputs, output_size=output_size)

# get the kernel as matrix
weight_as_matrix = conv_transpose_module.weight.transpose(0, 1).flatten(1)

# compute the output as a matrix, then reshape it into images
conv_t_via_unfold = torch.einsum("ci,nix->ncx", weight_as_matrix, inputs_unfolded)
conv_t_via_unfold = conv_t_via_unfold.reshape_as(conv_t)

# check
if torch.allclose(conv_t, conv_t_via_unfold):
    print("✔ Transpose convolution in PyTorch matches matrix multiplication approach.")
else:
    raise AssertionError("❌ Transpose convolutions don't match")
