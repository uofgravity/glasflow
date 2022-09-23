# -*- coding: utf-8 -*-
"""
Implementation of RealNVP.
"""
from glasflow.nflows.transforms.coupling import (
    AdditiveCouplingTransform,
)
import torch.nn.functional as F
from .coupling import CouplingFlow
from ..transforms import AffineCouplingTransform


class RealNVP(CouplingFlow):
    """Implementation of Real Non-Volume Preserving Flows.

    See: https://arxiv.org/abs/1605.08803

    Parameters
    ----------
    n_inputs : int
        Number of inputs
    n_transforms : int
        Number of transforms
    n_conditional_inputs: int
        Number of conditionals inputs
    n_neurons : int
        Number of neurons per residual block in each transform
    n_blocks_per_transform : int
        Number of residual blocks per transform
    batch_norm_within_blocks : bool
        Enable batch normalisation within each residual block
    batch_norm_between_transforms : bool
        Enable batch norm between transforms
    activation : function
        Activation function to use. Defaults to ReLU
    dropout_probability : float
        Amount of dropout to apply. 0 being no dropout and 1 being drop
        all connections
    linear_transform : str, {'permutation', 'lu', 'svd', None}
        Linear transform to apply before each coupling transform.
    distribution : :obj:`nflows.distribution.Distribution`
        Distribution object to use for that latent spae. If None, an n-d
        Gaussian is used.
    mask : Union[torch.Tensor, list, numpy.ndarray]
        Mask or array of masks to use to construct the flow. If not specified,
        an alternating binary mask will be used.
    volume_preserving : bool, optional
        If True use additive transforms that preserve volume.
    kwargs :
        Keyword arguments passed to either
        :py:obj:`nflows.transforms.coupling.AdditiveCouplingTransform` or
        :py:obj:`glasflow.transforms.coupling.AffineCouplingTransform`.
    """

    def __init__(
        self,
        n_inputs,
        n_transforms,
        n_conditional_inputs=None,
        n_neurons=32,
        n_blocks_per_transform=2,
        batch_norm_within_blocks=False,
        batch_norm_between_transforms=False,
        activation=F.relu,
        dropout_probability=0.0,
        linear_transform=None,
        distribution=None,
        mask=None,
        volume_preserving=False,
        **kwargs,
    ):
        if volume_preserving:
            transform_class = AdditiveCouplingTransform
        else:
            transform_class = AffineCouplingTransform
        super().__init__(
            transform_class,
            n_inputs,
            n_transforms,
            n_conditional_inputs=n_conditional_inputs,
            n_neurons=n_neurons,
            n_blocks_per_transform=n_blocks_per_transform,
            batch_norm_within_blocks=batch_norm_within_blocks,
            batch_norm_between_transforms=batch_norm_between_transforms,
            activation=activation,
            dropout_probability=dropout_probability,
            linear_transform=linear_transform,
            distribution=distribution,
            mask=mask,
            **kwargs,
        )
