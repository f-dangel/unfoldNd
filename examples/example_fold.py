"""How to use ``foldNd``. A comparison with ``torch.nn.Fold``."""

# imports, make this example deterministic
import torch

import unfoldNd

torch.manual_seed(0)

# random output of an im2col operation
inputs = torch.randn(64, 3 * 2 * 2, 12)
output_size = (4, 5)

# other module hyperparameters
kernel_size = 2
dilation = 1
padding = 0
stride = 1


# both modules accept the same arguments and perform the same operation
torch_module = torch.nn.Fold(
    output_size, kernel_size, dilation=dilation, padding=padding, stride=stride
)
lib_module = unfoldNd.FoldNd(
    output_size, kernel_size, dilation=dilation, padding=padding, stride=stride
)

# forward pass
torch_outputs = torch_module(inputs)
lib_outputs = lib_module(inputs)

# check
if torch.allclose(torch_outputs, lib_outputs):
    print("✔ Outputs of torch.nn.Fold and unfoldNd.FoldNd match.")
else:
    raise AssertionError("❌ Outputs don't match")
