# -*- coding: utf-8 -*-
"""
Implementation of Neural Spline Flows.

See: https://arxiv.org/abs/1906.04032
"""
from glasflow.nflows.transforms.coupling import (
    PiecewiseRationalQuadraticCouplingTransform,
)
from .coupling import CouplingFlow


class CouplingNSF(CouplingFlow):
    """Implementation of Neural Spline Flows using a coupling transform.

    See :obj:`glasflow.flow.coupling.CouplingFlow` for the complete list of
    parameters and methods.

    Parameters
    ----------
    args :
        Positional arguments passed to the parent class.
    num_bins : int
        Number of bins for the spline in each dimension.
    tail_type : {None, 'linear'}
        Type of tails to use outside the bounds on which the splines are
        defined.
    tail_bound : float
        Bound that defines the region over which the splines are defined.
        I.e. [-tail_bound, tail_bound]
    kwargs :
        Keyword arguments passed to the parent class.
    """

    def __init__(
        self, *args, num_bins=4, tail_type="linear", tail_bound=5.0, **kwargs
    ):
        transform_class = PiecewiseRationalQuadraticCouplingTransform
        super().__init__(
            transform_class,
            *args,
            num_bins=num_bins,
            tails=tail_type,
            tail_bound=tail_bound,
            **kwargs,
        )
