"""Problem settings for transpose test."""

from test.utils import get_available_devices, make_id

import torch

DEVICES = get_available_devices()
DEVICES_ID = [f"device={dev}" for dev in DEVICES]

PROBLEMS_2D = [
    {
        "seed": 0,
        "input_fn": lambda: torch.rand(2, 3, 50, 40),
        "out_channels": 2,
        "unfold_transpose_kwargs": {
            "kernel_size": 1,
        },
    },
]
PROBLEMS_2D_IDS = [make_id(problem) for problem in PROBLEMS_2D]
