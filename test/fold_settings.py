"""Problem settings for N-dimensional fold."""

from test.utils import get_available_devices, make_id

DEVICES = get_available_devices()
DEVICES_ID = [f"device={dev}" for dev in DEVICES]

PROBLEMS_2D = [
    {
        "seed": 0,
        "input_shape": (2, 3 * 2 * 2, 12),
        "fold_kwargs": {
            "output_size": (4, 5),
            "kernel_size": (2, 2),
        },
    },
]
PROBLEMS_2D_IDS = [make_id(problem) for problem in PROBLEMS_2D]
