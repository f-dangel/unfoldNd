"""Problem settings for transpose test."""

from test.utils import get_available_devices, make_id

import torch

DEVICES = get_available_devices()
DEVICES_ID = [f"device={dev}" for dev in DEVICES]

PROBLEMS_1D = [
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(3, 2, 20),
        "out_channels": 3,
        "groups": 1,
        "output_size": None,
        "unfold_transpose_kwargs": {
            "kernel_size": 2,
            "padding": 1,
        },
    },
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(3, 2, 111),
        "out_channels": 3,
        "groups": 1,
        "output_size": None,
        "unfold_transpose_kwargs": {
            "kernel_size": 2,
            "padding": 1,
            "stride": 2,
            "dilation": 2,
        },
    },
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(3, 3, 113),
        "out_channels": 6,
        "groups": 3,
        "output_size": None,
        "unfold_transpose_kwargs": {
            "kernel_size": 2,
            "padding": 2,
            "stride": 4,
            "dilation": 3,
        },
    },
]
PROBLEMS_1D_IDS = [make_id(problem) for problem in PROBLEMS_1D]

PROBLEMS_2D = [
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(2, 3, 50, 40),
        "out_channels": 2,
        "groups": 1,
        "output_size": None,
        "unfold_transpose_kwargs": {
            "kernel_size": 1,
        },
    },
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(3, 2, 10, 10),
        "out_channels": 3,
        "groups": 1,
        "output_size": None,
        "unfold_transpose_kwargs": {
            "kernel_size": 2,
            "padding": 1,
            "stride": 2,
            "dilation": 2,
        },
    },
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(1, 3, 16, 16),
        "out_channels": 6,
        "groups": 1,
        "output_size": None,
        "unfold_transpose_kwargs": {
            "kernel_size": 2,
            "padding": 2,
            "stride": 4,
            "dilation": 3,
        },
    },
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(3, 2, 11, 13),
        "out_channels": 6,
        "groups": 2,
        "output_size": None,
        "unfold_transpose_kwargs": {
            "kernel_size": 2,
            "padding": 1,
            "dilation": 2,
        },
    },
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(10, 8, 25, 50),
        "out_channels": 15,
        "groups": 1,
        "output_size": None,
        "unfold_transpose_kwargs": {
            "kernel_size": (3, 5),
            "padding": (4, 2),
            "stride": (2, 1),
            "dilation": (3, 1),
        },
    },
    # with nontrivial output_size, taken from
    # https://discuss.pytorch.org/t/the-output-size-of-convtranspose2d-differs-from-the-expected-output-size/1876/11 # noqa: B950
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(1, 3, 10, 10),
        "out_channels": 1,
        "groups": 1,
        "output_size": (21, 21),
        "unfold_transpose_kwargs": {
            "kernel_size": 2,
            "stride": 2,
        },
    },
]
PROBLEMS_2D_IDS = [make_id(problem) for problem in PROBLEMS_2D]

PROBLEMS_3D = [
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(3, 2, 7, 9, 9),
        "out_channels": 6,
        "groups": 1,
        "output_size": None,
        "unfold_transpose_kwargs": {
            "kernel_size": 2,
            "padding": 2,
            "stride": 2,
            "dilation": 2,
        },
    },
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(3, 2, 5, 13, 17),
        "out_channels": 3,
        "groups": 1,
        "output_size": None,
        "unfold_transpose_kwargs": {
            "kernel_size": 2,
            "padding": 1,
            "stride": 2,
            "dilation": 2,
        },
    },
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(3, 2, 23, 34, 55),
        "out_channels": 4,
        "groups": 2,
        "output_size": None,
        "unfold_transpose_kwargs": {
            "kernel_size": (2, 3, 4),
            "padding": (0, 1, 1),
            "stride": (1, 2, 2),
            "dilation": (1, 2, 2),
        },
    },
]
PROBLEMS_3D_IDS = [make_id(problem) for problem in PROBLEMS_3D]
