"""General utilities"""

from collections.abc import Iterable
from typing import Union

import torch


def get_torch_size(shape: Union[int, Iterable]) -> torch.Size:
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
    elif isinstance(shape, Iterable):
        shape = tuple(shape)
    return torch.Size(shape)
