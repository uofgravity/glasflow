from glasflow.nflows import transforms
import logging
import torch.nn.functional as F

from .base import Flow
from ..transforms.utils import get_scale_activation
from .. import USE_NFLOWS


logger = logging.getLogger(__name__)


class MaskedAutoregressiveFlow(Flow):
    """Base class for masked autoregressive flows.

    Parameters
    ----------
    transform_class : :obj:`nflows.transforms.autoregressive.AutoregressiveTransform`
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
        Amount of dropout to apply. 0 being no dropout and 1 being drop
        all connections
    use_random_permutations : bool
        If True, the order of the inputs is randomly permuted between
        transforms.
    use_random_masks : bool
        If True, random masks are used for the autoregressive transform.
    distribution : :obj:`nflows.distribution.Distribution`
        Distribution object to use for that latent spae. If None, an n-d
        Gaussian is used.
    kwargs :
        Keyword arguments passed to `transform_class` when is it initialised.
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
        use_random_permutations=False,
        use_random_masks=False,
        distribution=None,
        **kwargs,
    ):

        if use_random_permutations:
            permutation_constructor = transforms.RandomPermutation
        else:
            permutation_constructor = transforms.ReversePermutation

        layers = []
        for _ in range(n_transforms):
            layers.append(permutation_constructor(n_inputs))
            layers.append(
                transform_class(
                    features=n_inputs,
                    hidden_features=n_neurons,
                    context_features=n_conditional_inputs,
                    num_blocks=n_blocks_per_transform,
                    random_mask=use_random_masks,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=batch_norm_within_blocks,
                    **kwargs,
                )
            )
            if batch_norm_between_transforms:
                layers.append(transforms.BatchNorm(n_inputs))

        if distribution is None:
            from glasflow.nflows.distributions import StandardNormal

            distribution = StandardNormal([n_inputs])

        super().__init__(
            transform=transforms.CompositeTransform(layers),
            distribution=distribution,
        )


class MaskedAffineAutoregressiveFlow(MaskedAutoregressiveFlow):
    """Masked autoregressive flow with affine transforms.

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
    use_random_permutations : bool
        If True, the order of the inputs is randomly permuted between
        transforms.
    use_random_masks : bool
        If True, random masks are used for the autoregressive transform.
    distribution : :obj:`nflows.distribution.Distribution`
        Distribution object to use for that latent spae. If None, an n-d
        Gaussian is used.
    scale_activation : Optional[str, Callable]
        Activation for constraining the scale parameter.
    kwargs :
        Keyword arguments passed to `transform_class` when is it initialised.
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
        dropout_probability=0,
        use_random_permutations=False,
        use_random_masks=False,
        distribution=None,
        **kwargs,
    ):

        if USE_NFLOWS and kwargs.get("scale_activation", None) is not None:
            logger.error("nflows backend does not support scale activation")
        elif kwargs.get("scale_activation", None) is not None:
            kwargs["scale_activation"] = get_scale_activation(
                kwargs["scale_activation"]
            )

        super().__init__(
            transforms.autoregressive.MaskedAffineAutoregressiveTransform,
            n_inputs=n_inputs,
            n_transforms=n_transforms,
            n_conditional_inputs=n_conditional_inputs,
            n_neurons=n_neurons,
            n_blocks_per_transform=n_blocks_per_transform,
            batch_norm_within_blocks=batch_norm_within_blocks,
            batch_norm_between_transforms=batch_norm_between_transforms,
            activation=activation,
            dropout_probability=dropout_probability,
            use_random_permutations=use_random_permutations,
            use_random_masks=use_random_masks,
            distribution=distribution,
            **kwargs,
        )


class MaskedPiecewiseLinearAutoregressiveFlow(MaskedAutoregressiveFlow):
    """Masked autoregressive flow with piecewise linear splines.

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
    use_random_permutations : bool
        If True, the order of the inputs is randomly permuted between
        transforms.
    use_random_masks : bool
        If True, random masks are used for the autoregressive transform.
    distribution : :obj:`nflows.distribution.Distribution`
        Distribution object to use for that latent spae. If None, an n-d
        Gaussian is used.
    num_bins : int
        The number of bins.
    kwargs :
        Keyword arguments passed to `transform_class` when is it initialised.
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
        dropout_probability=0,
        use_random_permutations=False,
        use_random_masks=False,
        distribution=None,
        num_bins=10,
        **kwargs,
    ):
        super().__init__(
            transforms.autoregressive.MaskedPiecewiseLinearAutoregressiveTransform,
            n_inputs=n_inputs,
            n_transforms=n_transforms,
            n_conditional_inputs=n_conditional_inputs,
            n_neurons=n_neurons,
            n_blocks_per_transform=n_blocks_per_transform,
            batch_norm_within_blocks=batch_norm_within_blocks,
            batch_norm_between_transforms=batch_norm_between_transforms,
            activation=activation,
            dropout_probability=dropout_probability,
            use_random_permutations=use_random_permutations,
            use_random_masks=use_random_masks,
            distribution=distribution,
            num_bins=num_bins,
            **kwargs,
        )


class MaskedPiecewiseQuadraticAutoregressiveFlow(MaskedAutoregressiveFlow):
    """Masked autoregressive flow with piecewise quadratic splines.

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
    use_random_permutations : bool
        If True, the order of the inputs is randomly permuted between
        transforms.
    use_random_masks : bool
        If True, random masks are used for the autoregressive transform.
    distribution : :obj:`nflows.distribution.Distribution`
        Distribution object to use for that latent spae. If None, an n-d
        Gaussian is used.
    num_bins : int
        The number of bins.
    kwargs :
        Keyword arguments passed to `transform_class` when is it initialised.
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
        dropout_probability=0,
        use_random_permutations=False,
        use_random_masks=False,
        distribution=None,
        num_bins=10,
        **kwargs,
    ):
        super().__init__(
            transforms.autoregressive.MaskedPiecewiseQuadraticAutoregressiveTransform,
            n_inputs=n_inputs,
            n_transforms=n_transforms,
            n_conditional_inputs=n_conditional_inputs,
            n_neurons=n_neurons,
            n_blocks_per_transform=n_blocks_per_transform,
            batch_norm_within_blocks=batch_norm_within_blocks,
            batch_norm_between_transforms=batch_norm_between_transforms,
            activation=activation,
            dropout_probability=dropout_probability,
            use_random_permutations=use_random_permutations,
            use_random_masks=use_random_masks,
            distribution=distribution,
            num_bins=num_bins,
            **kwargs,
        )


class MaskedPiecewiseCubicAutoregressiveAutoregressiveFlow(
    MaskedAutoregressiveFlow
):
    """Masked autoregressive flow with piecewise cubic splines.

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
    use_random_permutations : bool
        If True, the order of the inputs is randomly permuted between
        transforms.
    use_random_masks : bool
        If True, random masks are used for the autoregressive transform.
    distribution : :obj:`nflows.distribution.Distribution`
        Distribution object to use for that latent spae. If None, an n-d
        Gaussian is used.
    num_bins : int
        The number of bins.
    kwargs :
        Keyword arguments passed to `transform_class` when is it initialised.
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
        dropout_probability=0,
        use_random_permutations=False,
        use_random_masks=False,
        distribution=None,
        num_bins=10,
        **kwargs,
    ):
        super().__init__(
            transforms.autoregressive.MaskedPiecewiseCubicAutoregressiveTransform,
            n_inputs=n_inputs,
            n_transforms=n_transforms,
            n_conditional_inputs=n_conditional_inputs,
            n_neurons=n_neurons,
            n_blocks_per_transform=n_blocks_per_transform,
            batch_norm_within_blocks=batch_norm_within_blocks,
            batch_norm_between_transforms=batch_norm_between_transforms,
            activation=activation,
            dropout_probability=dropout_probability,
            use_random_permutations=use_random_permutations,
            use_random_masks=use_random_masks,
            distribution=distribution,
            num_bins=num_bins,
            **kwargs,
        )


class MaskedPiecewiseRationalQuadraticAutoregressiveFlow(
    MaskedAutoregressiveFlow
):
    """Masked autoregressive flow with piecewise rational quadratic splines.

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
    use_random_permutations : bool
        If True, the order of the inputs is randomly permuted between
        transforms.
    use_random_masks : bool
        If True, random masks are used for the autoregressive transform.
    distribution : :obj:`nflows.distribution.Distribution`
        Distribution object to use for that latent spae. If None, an n-d
        Gaussian is used.
    num_bins : int
        The number of bins.
    kwargs :
        Keyword arguments passed to `transform_class` when is it initialised.
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
        dropout_probability=0,
        use_random_permutations=False,
        use_random_masks=False,
        distribution=None,
        num_bins=10,
        **kwargs,
    ):
        super().__init__(
            transforms.autoregressive.MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
            n_inputs=n_inputs,
            n_transforms=n_transforms,
            n_conditional_inputs=n_conditional_inputs,
            n_neurons=n_neurons,
            n_blocks_per_transform=n_blocks_per_transform,
            batch_norm_within_blocks=batch_norm_within_blocks,
            batch_norm_between_transforms=batch_norm_between_transforms,
            activation=activation,
            dropout_probability=dropout_probability,
            use_random_permutations=use_random_permutations,
            use_random_masks=use_random_masks,
            distribution=distribution,
            num_bins=num_bins,
            **kwargs,
        )
