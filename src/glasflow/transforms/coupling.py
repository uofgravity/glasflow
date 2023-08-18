# -*- coding: utf-8 -*-
"""Alternative implementations of coupling transforms"""
import logging
import warnings

from glasflow.nflows.transforms.coupling import (
    AffineCouplingTransform as BaseAffineCouplingTransform,
)
import torch.nn.functional as F

from .utils import get_scale_activation
from .. import USE_NFLOWS


logger = logging.getLogger(__name__)


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
        scale_activation="nflows_general",
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

        try:
            super().__init__(
                mask,
                transform_net_create_fn,
                unconditional_transform=unconditional_transform,
                scale_activation=scale_activation,
                **kwargs,
            )
        except TypeError as e:
            if USE_NFLOWS:
                logger.error(
                    (
                        f"Could not initialise transform with with error: {e}. "
                        "The version of `nflows` being used may not support "
                        "`scale_activation`. Trying without `scale_activation`."
                        " Full traceback:"
                    ),
                    exc_info=True,
                )
                super().__init__(
                    mask,
                    transform_net_create_fn,
                    unconditional_transform=unconditional_transform,
                    **kwargs,
                )
                logger.warning(
                    "Using affine coupling transform without "
                    "`scale_activation`, this is not recommended!"
                )
            else:
                raise e
