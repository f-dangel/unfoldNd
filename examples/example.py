"""How to use ``unfoldNd``. A comparison with ``torch.nn.Unfold``."""

# imports, make this example deterministic
import torch

import unfoldNd

torch.manual_seed(0)

# random batched RGB 32x32 image-shaped input tensor of batch size 64
inputs = torch.randn((64, 3, 32, 32))

# module hyperparameters
kernel_size = 3
dilation = 1
padding = 1
stride = 2

# both modules accept the same arguments and perform the same operation
torch_module = torch.nn.Unfold(
    kernel_size, dilation=dilation, padding=padding, stride=stride
)
lib_module = unfoldNd.UnfoldNd(
    kernel_size, dilation=dilation, padding=padding, stride=stride
)

# forward pass
torch_outputs = torch_module(inputs)
lib_outputs = lib_module(inputs)

# check
if torch.allclose(torch_outputs, lib_outputs):
    print("✔ Outputs of torch.nn.Unfold and unfoldNd.UnfoldNd match.")
else:
    raise AssertionError("❌ Outputs don't match")
