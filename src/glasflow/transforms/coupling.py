# -*- coding: utf-8 -*-
"""Alternative implementations of coupling transforms"""
from glasflow.nflows.transforms.coupling import (
    AffineCouplingTransform as BaseAffineCouplingTransform,
)
import torch.nn.functional as F


class AffineCouplingTransform(BaseAffineCouplingTransform):
    """Modified affine coupling transform that has different scaling options.

    Adds the option to use different ranges for the scaling parameters. In
    `nflows` the scale is limited to [0, 1.001]. This method adds the `'wide'`
    option where the scale is limited to [0, 3].
    """

    _allowed_scaling_methods = ["nflows", "wide"]

    def __init__(
        self,
        mask,
        transform_net_create_fn,
        unconditional_transform=None,
        scaling_method="nflows",
    ):
        super().__init__(
            mask,
            transform_net_create_fn,
            unconditional_transform=unconditional_transform,
        )

        if scaling_method in self._allowed_scaling_methods:
            self.scaling_method = scaling_method
        else:
            raise ValueError(
                f"Invalid scaling method: {scaling_method}. "
                f"Choose from: {self._allowed_scaling_methods}."
            )

    def _scale_and_shift_wide(self, transform_params):
        unconstrained_scale = transform_params[
            :, self.num_transform_features :, ...
        ]
        shift = transform_params[:, : self.num_transform_features, ...]
        scale = (F.softplus(unconstrained_scale) + 1e-3).clamp(0, 3)
        return scale, shift

    def _scale_and_shift(self, transform_params):
        if self.scaling_method == "nflows":
            return super()._scale_and_shift(transform_params)
        elif self.scaling_method == "wide":
            return self._scale_and_shift_wide(transform_params)
        else:
            raise RuntimeError(
                f"Unknown scaling method: {self.scaling_method}"
            )
