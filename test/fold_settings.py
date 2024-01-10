"""Problem settings for N-dimensional fold."""

from test.utils import get_available_devices, make_id

import torch

DEVICES = get_available_devices()
DEVICES_ID = [f"device={dev}" for dev in DEVICES]

PROBLEMS_2D = [
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(2, 3 * 2 * 2, 12),
        "fold_kwargs": {
            "output_size": (4, 5),
            "kernel_size": (2, 2),
        },
    },
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(2, 3 * 2 * 2, 5 * 9),
        "fold_kwargs": {
            "output_size": (4, 8),
            "kernel_size": 2,
            "padding": 1,
        },
        "id": "bug30-fold-with-padding",
    },
]
PROBLEMS_2D_IDS = [make_id(problem) for problem in PROBLEMS_2D]

UNSUPPORTED_ARGS_PROBLEMS = [
    # output size is integer
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(2, 3 * 2 * 2, 12),
        "fold_kwargs": {
            "output_size": 4,
            "kernel_size": (2, 2),
        },
    },
]
UNSUPPORTED_ARGS_PROBLEMS_IDS = [
    make_id(problem) for problem in UNSUPPORTED_ARGS_PROBLEMS
]

PRECISION_PROBLEMS_2D = [
    # out-of-bounds error because float index is rounded up
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(1, 1 * 2 * 2, 25),
        "fold_kwargs": {
            # > smallest int which is exact as float32, 2 ** 24 = 116777217
            # (see https://stackoverflow.com/q/27207149 for details)
            "output_size": (2**12 + 2, 2**12 + 2),
            "kernel_size": (2, 2),
            "stride": 2**10,
        },
    },
    # wrong result due to wrong float → long conversion
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(1, 1 * 2 * 2, 25),
        "fold_kwargs": {
            # > smallest int which is exact as float32, 2 ** 24 = 116777217
            # (see https://stackoverflow.com/q/27207149 for details)
            "output_size": (5000, 5000),
            "kernel_size": (2, 2),
            "stride": 1000,
        },
    },
]
PRECISION_PROBLEMS_2D_IDS = [make_id(problem) for problem in PRECISION_PROBLEMS_2D]

# Settings must satisfy ``fold(unfold(input)) = input``
PROBLEMS_INVERSE = [
    # 1d basic
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(2, 3, 50),
        "unfold_kwargs": {
            "kernel_size": 2,
            "stride": 2,
        },
    },
    # 1d with dilation
    {
        "seed": 1,
        "input_fn": lambda: torch.rand(2, 3, 50),
        "unfold_kwargs": {
            "kernel_size": 5,
            "dilation": 10,
            "stride": 1,
        },
    },
    # 1d with padding
    {
        "seed": 2,
        "input_fn": lambda: torch.rand(2, 3, 50),
        "unfold_kwargs": {
            "kernel_size": 4,
            "padding": 2,
            "stride": 4,
        },
    },
    # 1d with padding and dilation
    {
        "seed": 3,
        "input_fn": lambda: torch.rand(2, 3, 50),
        "unfold_kwargs": {
            "kernel_size": 3,
            "dilation": 18,
            "stride": 1,
            "padding": 2,
        },
    },
    # 2d basic
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(2, 3, 50, 40),
        "unfold_kwargs": {
            "kernel_size": 2,
            "stride": 2,
        },
    },
    # 2d with dilation
    {
        "seed": 1,
        "input_fn": lambda: torch.rand(2, 3, 50, 40),
        "unfold_kwargs": {
            "kernel_size": (5, 2),
            "stride": (1, 1),
            "dilation": (10, 20),
        },
    },
    # 2d with padding
    {
        "seed": 2,
        "input_fn": lambda: torch.rand(2, 3, 50, 40),
        "unfold_kwargs": {
            "kernel_size": (4, 3),
            "padding": (2, 1),
            "stride": (4, 3),
        },
    },
    # 2d with padding and dilation
    {
        "seed": 3,
        "input_fn": lambda: torch.rand(2, 3, 50, 40),
        "unfold_kwargs": {
            "kernel_size": (3, 2),
            "dilation": (18, 21),
            "stride": (1, 1),
            "padding": (2, 1),
        },
    },
    # 3d basic
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(2, 3, 50, 40, 30),
        "unfold_kwargs": {
            "kernel_size": 2,
            "stride": 2,
        },
    },
    # 3d with dilation
    {
        "seed": 1,
        "input_fn": lambda: torch.rand(2, 3, 50, 40, 30),
        "unfold_kwargs": {
            "kernel_size": (5, 2, 6),
            "stride": (1, 1, 1),
            "dilation": (10, 20, 5),
        },
    },
    # 3d with padding
    {
        "seed": 2,
        "input_fn": lambda: torch.rand(2, 3, 50, 40, 30),
        "unfold_kwargs": {
            "kernel_size": (4, 3, 6),
            "padding": (2, 1, 3),
            "stride": (4, 3, 6),
        },
    },
    # 3d with padding and dilation
    {
        "seed": 3,
        "input_fn": lambda: torch.rand(2, 3, 50, 40, 30),
        "unfold_kwargs": {
            "kernel_size": (3, 2, 4),
            "dilation": (18, 21, 9),
            "stride": (1, 1, 1),
            "padding": (2, 1, 3),
        },
    },
]
PROBLEMS_INVERSE_IDS = [make_id(problem) for problem in PROBLEMS_INVERSE]
