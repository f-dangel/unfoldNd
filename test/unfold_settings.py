"""Problem settings for test."""

from test.utils import get_available_devices, make_id

import torch

DEVICES = get_available_devices()
DEVICES_ID = [f"device={dev}" for dev in DEVICES]

PROBLEMS_1D = [
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(2, 3, 50),
        "unfold_kwargs": {
            "kernel_size": 1,
        },
    },
    {
        "seed": 1,
        "input_fn": lambda: torch.rand(2, 3, 50),
        "unfold_kwargs": {
            "kernel_size": 2,
        },
    },
    {
        "seed": 2,
        "input_fn": lambda: torch.rand(2, 3, 50),
        "unfold_kwargs": {
            "kernel_size": 3,
        },
    },
    {
        "seed": 3,
        "input_fn": lambda: torch.rand(2, 3, 50),
        "unfold_kwargs": {
            "kernel_size": 3,
            "dilation": 2,
        },
    },
    {
        "seed": 4,
        "input_fn": lambda: torch.rand(2, 3, 50),
        "unfold_kwargs": {
            "kernel_size": 3,
            "dilation": 2,
            "padding": 1,
        },
    },
    {
        "seed": 5,
        "input_fn": lambda: torch.rand(2, 3, 50),
        "unfold_kwargs": {
            "kernel_size": 3,
            "dilation": 2,
            "padding": 1,
            "stride": 2,
        },
    },
]
PROBLEMS_1D_IDS = [make_id(problem) for problem in PROBLEMS_1D]

PROBLEMS_2D = [
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(2, 3, 50, 40),
        "unfold_kwargs": {
            "kernel_size": 1,
        },
    },
    {
        "seed": 1,
        "input_fn": lambda: torch.rand(2, 3, 50, 40),
        "unfold_kwargs": {
            "kernel_size": 2,
        },
    },
    {
        "seed": 2,
        "input_fn": lambda: torch.rand(2, 3, 50, 40),
        "unfold_kwargs": {
            "kernel_size": (3, 2),
        },
    },
    {
        "seed": 3,
        "input_fn": lambda: torch.rand(2, 3, 50, 40),
        "unfold_kwargs": {
            "kernel_size": (3, 2),
            "dilation": 2,
        },
    },
    {
        "seed": 4,
        "input_fn": lambda: torch.rand(2, 3, 50, 40),
        "unfold_kwargs": {
            "kernel_size": (3, 2),
            "dilation": 2,
            "padding": 1,
        },
    },
    {
        "seed": 5,
        "input_fn": lambda: torch.rand(2, 3, 50, 40),
        "unfold_kwargs": {
            "kernel_size": (3, 2),
            "dilation": 2,
            "padding": 1,
            "stride": 2,
        },
    },
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(2, 3, 50, 40, dtype=torch.float64),
        "unfold_kwargs": {
            "kernel_size": 1,
        },
        "id": "bug-float-64-input",
    },
]
PROBLEMS_2D_IDS = [make_id(problem) for problem in PROBLEMS_2D]


PROBLEMS_3D = [
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(2, 3, 50, 40, 30),
        "unfold_kwargs": {
            "kernel_size": 1,
        },
        "out_channels": 1,
    },
    {
        "seed": 1,
        "input_fn": lambda: torch.rand(2, 3, 50, 40, 30),
        "unfold_kwargs": {
            "kernel_size": 2,
        },
        "out_channels": 2,
    },
    {
        "seed": 2,
        "input_fn": lambda: torch.rand(2, 3, 50, 40, 30),
        "unfold_kwargs": {
            "kernel_size": (4, 3, 2),
        },
        "out_channels": 4,
    },
    {
        "seed": 3,
        "input_fn": lambda: torch.rand(2, 3, 50, 40, 30),
        "unfold_kwargs": {
            "kernel_size": (4, 3, 2),
            "dilation": 2,
        },
        "out_channels": 5,
    },
    {
        "seed": 4,
        "input_fn": lambda: torch.rand(2, 3, 50, 40, 30),
        "unfold_kwargs": {
            "kernel_size": (4, 3, 2),
            "dilation": 2,
            "padding": 1,
        },
        "out_channels": 10,
    },
    {
        "seed": 5,
        "input_fn": lambda: torch.rand(2, 3, 50, 40, 30),
        "unfold_kwargs": {
            "kernel_size": (4, 3, 2),
            "dilation": 2,
            "padding": 1,
            "stride": 2,
        },
        "out_channels": 10,
    },
]
PROBLEMS_3D_IDS = [make_id(problem) for problem in PROBLEMS_3D]
