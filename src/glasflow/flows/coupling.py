# -*- coding: utf-8 -*-
from nflows.nn.nets import ResidualNet
from nflows import transforms
from nflows.transforms.coupling import CouplingTransform
import torch
import torch.nn.functional as F

from .base import Flow


class CouplingFlow(Flow):
    """Base class for coupling transform based flows.

    Uses an alternating binary mask and residual neural network. Others
    settings can be configured. See parameters for details.

    Parameters
    ----------
    transform_class : :obj:`nflows.transforms.coupling.CouplingTransform`
        Class that inherits from `CouplingTransform` and implements the 
        actual transformation.
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
        Ammount of dropout to apply. 0 being no dropout and 1 being drop
        all connections
    linear_transform : str, {'permutation', 'lu', 'svd', None}
        Linear transform to apply before each coupling transform.
    distribution : :obj:`nflows.distribution.Distribtion`
        Distribution object to use for that latent spae. If None, an n-d 
        Gaussian is used.
    kwargs : 
        Keyword arguments passed to `transform_class` when is it initialsed.
    """
    def __init__(
        self,
        transform_class,
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
        **kwargs
    ):

        if not issubclass(transform_class, CouplingTransform):
            raise RuntimeError(
                'Transform class does not inherit from `CouplingTransform`'
            ) 

        def create_net(n_in, n_out):
            return ResidualNet(
                n_in,
                n_out,
                hidden_features=n_neurons,
                context_features=n_conditional_inputs,
                num_blocks=n_blocks_per_transform,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_blocks
                )

        def create_linear_transform():
            if linear_transform == 'permutation':
                return transforms.RandomPermutation(features=n_inputs)
            elif linear_transform == 'lu':
                return transforms.CompositeTransform([
                    transforms.RandomPermutation(features=n_inputs),
                    transforms.LULinear(n_inputs, identity_init=True,
                                        using_cache=False)
                ])
            elif linear_transform == 'svd':
                return transforms.CompositeTransform([
                    transforms.RandomPermutation(features=n_inputs),
                    transforms.SVDLinear(n_inputs, num_householder=10,
                                         identity_init=True)
                ])
            else:
                raise ValueError(
                    f'Unknown linear transform: {linear_transform}.'
                )

        def create_transform(mask):
            return transform_class(
                mask=mask,
                transform_net_create_fn=create_net,
                **kwargs
            )

        mask = torch.ones(n_inputs)
        mask[::2] = -1
				
        transforms_list = []

        for _ in range(n_transforms):
            if linear_transform is not None:
                transforms_list.append(create_linear_transform())
            transforms_list.append(create_transform(mask))
            mask *= -1
            if batch_norm_between_transforms:
                transforms_list.append(transforms.BatchNorm(n_inputs))

        
        if distribution is None:
            from nflows.distributions import StandardNormal
            distribution = StandardNormal([n_inputs])

        super().__init__(
            transform=transforms.CompositeTransform(transforms_list),
            distribution=distribution,
        )
