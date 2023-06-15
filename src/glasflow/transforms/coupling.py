# -*- coding: utf-8 -*-
"""Alternative implementations of coupling transforms"""
import warnings

from glasflow.nflows.transforms.coupling import (
    AffineCouplingTransform as BaseAffineCouplingTransform,
)
import torch.nn.functional as F

from .utils import get_scale_activation


class AffineCouplingTransform(BaseAffineCouplingTransform):
    """Modified affine coupling transform that includes predefined scale
    activations.

    Adds the option specify `scale_activation` as a string which is passed to
    `get_scale_activation` to get the corresponding function. Also supports
    specifying the function instead of string.
    """

    def __init__(
        self,
        mask,
        transform_net_create_fn,
        unconditional_transform=None,
        scaling_method=None,
        scale_activation=None,
        **kwargs,
    ):
        if scaling_method is not None:
            warnings.warn(
                (
                    "scaling_method is deprecated and will be removed in a "
                    "future release. Use `scale_activation` instead."
                ),
                FutureWarning,
            )
            scale_activation = scaling_method

        scale_activation = get_scale_activation(scale_activation)

        super().__init__(
            mask,
            transform_net_create_fn,
            unconditional_transform=unconditional_transform,
            scale_activation=scale_activation,
            **kwargs,
        )
