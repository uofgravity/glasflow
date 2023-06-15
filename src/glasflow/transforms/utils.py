"""Utilities for the transforms submodule"""
from typing import Callable, Union
import torch
import torch.nn.functional as F


SCALE_ACTIVATIONS = dict(
    nflows=lambda x: torch.sigmoid(x + 2) + 1e-3,
    nflows_general=lambda x: (F.softplus(x) + 1e-3).clamp(0, 3),
    wide=lambda x: (F.softplus(x) + 1e-3).clamp(0, 3),
    log1=lambda x: torch.exp(2 * (torch.sigmoid(x) - 0.5)),
    log2=lambda x: torch.exp(4 * (torch.sigmoid(x) - 0.5)),
    log3=lambda x: torch.exp(6 * (torch.sigmoid(x) - 0.5)),
)


def get_scale_activation(activation: Union[str, Callable]) -> Callable:
    """Get the scale activation function.

    If `activation` is not a string, then it is returned.
    """
    if not isinstance(activation, str):
        return activation
    elif activation in SCALE_ACTIVATIONS:
        return SCALE_ACTIVATIONS.get(activation)
    else:
        raise ValueError(
            f"Unknown activation: {activation}. "
            f"Choose from: {SCALE_ACTIVATIONS}"
        )
