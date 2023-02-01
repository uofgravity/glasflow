# -*- coding: utf-8 -*-
"""
Implementation of Neural Spline Flows.

See: https://arxiv.org/abs/1906.04032
"""
from glasflow.nflows.transforms.coupling import (
    PiecewiseRationalQuadraticCouplingTransform,
)
import torch
import torch.nn.functional as F
from .coupling import CouplingFlow


class CouplingNSF(CouplingFlow):
    """Implementation of Neural Spline Flows using a coupling transform.

    Supports use of a uniform distribution for the latent space. This
    automatically disables the tails and sets the bounds to [0, 1).

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
    num_bins : int
        Number of bins for the spline in each dimension.
    tail_type : {None, 'linear'}
        Type of tails to use outside the bounds on which the splines are
        defined.
    tail_bound : float
        Bound that defines the region over which the splines are defined.
        I.e. [-tail_bound, tail_bound]
    kwargs :
        Keyword arguments passed to the transform when is it initialised.
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
        num_bins=4,
        tail_type="linear",
        tail_bound=5.0,
        **kwargs,
    ):
        transform_class = PiecewiseRationalQuadraticCouplingTransform

        if distribution == "uniform":
            from ..distributions import MultivariateUniform

            tail_bound = 1.0
            tail_type = None
            distribution = MultivariateUniform(
                low=torch.Tensor(n_inputs * [0.0]),
                high=torch.Tensor(n_inputs * [1.0]),
            )
            batch_norm_between_transforms = False

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
            num_bins=num_bins,
            tails=tail_type,
            tail_bound=tail_bound,
            **kwargs,
        )
