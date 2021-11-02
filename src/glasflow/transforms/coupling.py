# -*- coding: utf-8 -*-
"""Alternative implementations of coupling transforms"""
from nflows.transforms.coupling import (
    AffineCouplingTransform as BaseAffineCouplingTransform,
    CouplingTransform,
)
from nflows.utils import torchutils
import torch
from torch import nn
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


class AffineCouplingTransformLogScale(CouplingTransform):
    """Version of AffineCouplingTransform that predicts the log-scale.
    
    Allows for a learnable scale parameter the rescales the log-scale as
    described here: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html#Coupling-layers
    """

    def __init__(
        self,
        mask,
        transform_net_create_fn,
        unconditional_transform=None,
    ):
        super().__init__(
            mask,
            transform_net_create_fn,
            unconditional_transform=unconditional_transform,
        )

        self.scaling_factor = nn.Parameter(
            torch.ones(self.num_transform_features)
        )

    def _transform_dim_multiplier(self):
        return 2

    def _scale_and_shift(self, transform_params):
        log_scale = transform_params[:, self.num_transform_features:, ...]
        shift = transform_params[:, : self.num_transform_features, ...]
        scale_factor = self.scaling_factor.exp()
        log_scale = torch.tanh(log_scale / scale_factor) * scale_factor
        return log_scale, shift

    def _coupling_transform_forward(self, inputs, transform_params):
        log_scale, shift = self._scale_and_shift(transform_params)
        outputs = inputs * torch.exp(log_scale) + shift
        logabsdet = torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _coupling_transform_inverse(self, inputs, transform_params):
        log_scale, shift = self._scale_and_shift(transform_params)
        outputs = (inputs - shift) / torch.exp(log_scale)
        logabsdet = -torchutils.sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet
