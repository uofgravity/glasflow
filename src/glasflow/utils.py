"""General utilities"""
from typing import Union

import torch


def get_torch_size(shape: Union[int, tuple]) -> torch.Size:
    """Get a torch size from a more flexible input shape.

    Parameters
    ----------
    shape
        The shape to convert to an instance of `torch.Size`.

    Returns
    -------
    The torch size
    """
    if isinstance(shape, int):
        shape = (shape,)
    return torch.Size(shape)
